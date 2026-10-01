//! The MCP tool surface over real HTTP, with bearer auth (T021).
//!
//! The transport is a second front end over the T020 handlers, so the tool
//! assertions are the T020 ones run again over the wire rather than a second
//! set: what this file is really testing is that the transport carries what
//! the surface produced, and that nothing reaches a handler without a token.
//!
//! The client is written against a `TcpStream` rather than a library, for the
//! same reason the stdio tests speak JSON-RPC by hand: a status code or a
//! header that is right in Rust and wrong on the wire is wrong, and an HTTP
//! client that papers over a 401 would hide the property this task exists to
//! establish.

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

use kb::mcp::{ToolOutcome, ToolSpec, ToolSurface};
use kb::mcp_http::{HttpConfig, HttpServer};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// The token every test in this file authenticates with.
const TOKEN: &str = "a-shared-secret-for-the-operator-network";

// ── a small HTTP/1.1 client ─────────────────────────────────────────────

/// One HTTP response, as it arrived.
struct HttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl HttpResponse {
    /// The first value of `name`, matched case-insensitively as HTTP requires.
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    /// The JSON-RPC payload, whether it arrived as JSON or as one SSE event.
    fn payload(&self) -> Result<Value, Box<dyn std::error::Error>> {
        let text = self.body.trim();
        if let Some(rest) = text.strip_prefix("event:") {
            let data = rest
                .lines()
                .find_map(|line| line.trim().strip_prefix("data:"))
                .ok_or("an SSE response carried no data line")?;
            return Ok(serde_json::from_str(data.trim())?);
        }
        Ok(serde_json::from_str(text)?)
    }
}

/// POST a JSON-RPC message to the server's `/mcp` endpoint.
fn post(
    addr: SocketAddr,
    token: Option<&str>,
    session: Option<&str>,
    body: &Value,
) -> Result<HttpResponse, Box<dyn std::error::Error>> {
    let payload = body.to_string();
    let mut request = format!(
        "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\nContent-Length: {}\r\n",
        payload.len()
    );
    if let Some(token) = token {
        let _ = write!(request, "Authorization: {token}\r\n");
    }
    if let Some(session) = session {
        let _ = write!(request, "Mcp-Session-Id: {session}\r\n");
    }
    request.push_str("Connection: close\r\n\r\n");
    request.push_str(&payload);

    let mut stream = TcpStream::connect(addr)?;
    stream.write_all(request.as_bytes())?;
    stream.flush()?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    parse_response(&String::from_utf8_lossy(&raw))
}

/// Parse a response, de-chunking a chunked body.
fn parse_response(raw: &str) -> Result<HttpResponse, Box<dyn std::error::Error>> {
    let (head, body) = raw
        .split_once("\r\n\r\n")
        .ok_or_else(|| format!("no header/body boundary in: {raw:?}"))?;
    let mut lines = head.lines();
    let status = lines
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or_else(|| format!("no status line in: {head:?}"))?
        .parse::<u16>()?;
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim().to_owned(), value.trim().to_owned()))
        .collect();
    let chunked = headers
        .iter()
        .any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding") && v.contains("chunked"));
    let body = if chunked {
        dechunk(body)
    } else {
        body.to_owned()
    };
    Ok(HttpResponse {
        status,
        headers,
        body,
    })
}

/// Reassemble a chunked body.
fn dechunk(body: &str) -> String {
    let mut out = String::new();
    let mut rest = body;
    while let Some((size, tail)) = rest.split_once("\r\n") {
        let Ok(len) = usize::from_str_radix(size.trim(), 16) else {
            break;
        };
        if len == 0 || tail.len() < len {
            out.push_str(tail.get(..tail.len().min(len)).unwrap_or_default());
            break;
        }
        out.push_str(tail.get(..len).unwrap_or_default());
        rest = tail.get(len.saturating_add(2)..).unwrap_or_default();
    }
    out
}

// ── a server running in the background ──────────────────────────────────

/// A server listening on a loopback port, with its own runtime.
struct Served {
    addr: SocketAddr,
    shutdown: kb::mcp_http::Shutdown,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Served {
    /// Start `surface` on an operating-system-chosen loopback port.
    fn start<S: ToolSurface + Send + Sync + 'static>(
        surface: S,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = HttpConfig::new("127.0.0.1:0".parse::<SocketAddr>()?, TOKEN.to_owned());
        let (ready, started) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(e) => {
                    drop(ready.send(Err(e.to_string())));
                    return;
                }
            };
            runtime.block_on(async move {
                let server = match HttpServer::bind(config).await {
                    Ok(server) => server,
                    Err(e) => {
                        drop(ready.send(Err(e.to_string())));
                        return;
                    }
                };
                if ready
                    .send(Ok((server.address(), server.shutdown())))
                    .is_err()
                {
                    return;
                }
                drop(server.serve(surface).await);
            });
        });
        let (addr, shutdown) = started.recv()??;
        Ok(Self {
            addr,
            shutdown,
            thread: Some(thread),
        })
    }

    /// Complete the initialize handshake and return the session id, if the
    /// server issued one.
    fn initialize(&self) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let response = post(
            self.addr,
            Some(&format!("Bearer {TOKEN}")),
            None,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": { "name": "kb-tests", "version": "0" },
                },
            }),
        )?;
        assert_eq!(response.status, 200, "initialize failed: {}", response.body);
        let session = response.header("mcp-session-id").map(str::to_owned);
        if session.is_some() {
            post(
                self.addr,
                Some(&format!("Bearer {TOKEN}")),
                session.as_deref(),
                &json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
            )?;
        }
        Ok(session)
    }

    /// Send a JSON-RPC request on an initialized session.
    fn request(
        &self,
        session: Option<&str>,
        method: &str,
        params: &Value,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let response = post(
            self.addr,
            Some(&format!("Bearer {TOKEN}")),
            session,
            &json!({ "jsonrpc": "2.0", "id": 2, "method": method, "params": params }),
        )?;
        assert_eq!(response.status, 200, "{method} failed: {}", response.body);
        response.payload()
    }

    /// Call a tool, returning its text and whether it reported failure.
    fn call(
        &self,
        session: Option<&str>,
        name: &str,
        arguments: &Value,
    ) -> Result<(String, bool), Box<dyn std::error::Error>> {
        let response = self.request(
            session,
            "tools/call",
            &json!({ "name": name, "arguments": arguments }),
        )?;
        let result = response
            .get("result")
            .ok_or_else(|| format!("no result in {response}"))?;
        let failed = result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let text = result
            .get("content")
            .and_then(Value::as_array)
            .and_then(|blocks| blocks.first())
            .and_then(|block| block.get("text"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("no text content in {result}"))?
            .to_owned();
        Ok((text, failed))
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        self.shutdown.stop();
        if let Some(thread) = self.thread.take() {
            drop(thread.join());
        }
    }
}

/// A database of three notes, one of which links to another.
fn populated()
-> Result<(tempfile::TempDir, std::path::PathBuf, std::path::PathBuf), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let store_path = dir.path().join("store");
    let index_path = dir.path().join("index.db");
    let store = kb::store::GitBlobStore::open_or_init(&store_path)?;
    let index = kb::index::Index::open_for_rebuild(&index_path)?;
    for (id, body) in [
        (
            "alpha",
            "* Ownership in Rust\n\nthe borrow checker and its rules\n",
        ),
        (
            "beta",
            "* Embedding models\n\nvectors for retrieval over notes\n",
        ),
        (
            "gamma",
            "* Retrieval notes\n\nsee [[id:alpha]] for the ownership piece\n",
        ),
    ] {
        let document = kb::parser::parse_document(body)?;
        let options = kb::write::WriteOptions::note(id)?;
        kb::write::put_record(&store, &index, id, &document, &options)?;
    }
    Ok((dir, store_path, index_path))
}

/// A server over the real tool surface.
fn served(store: &Path, index: &Path) -> Result<Served, Box<dyn std::error::Error>> {
    Served::start(kb::mcp::KbTools::with_paths(store, index))
}

// ── the T020 assertions, over HTTP ──────────────────────────────────────

#[test]
fn the_same_four_tools_are_offered_over_http() -> TestResult {
    let (_dir, store, index) = populated()?;
    let server = served(&store, &index)?;
    let session = server.initialize()?;
    let response = server.request(session.as_deref(), "tools/list", &json!({}))?;
    let names: Vec<&str> = response
        .get("result")
        .and_then(|result| result.get("tools"))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no tools in {response}"))?
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect();
    assert_eq!(names, vec!["search", "context", "get", "put"]);
    Ok(())
}

#[test]
fn a_client_asking_for_a_newer_revision_is_answered_with_one_the_server_implements() -> TestResult {
    // rmcp knows 2026-07-28 and would echo it; this server does not produce
    // that revision's result shapes, and a client that was promised it rejects
    // tools/list. Observed with Claude Code 2.1.240 on 2026-08-22.
    let (_dir, store, index) = populated()?;
    let server = served(&store, &index)?;
    for (asked, answered) in [
        ("2026-07-28", "2025-11-25"),
        ("2025-11-25", "2025-11-25"),
        ("2025-06-18", "2025-06-18"),
        ("9999-01-01", "2025-11-25"),
    ] {
        let response = post(
            server.addr,
            Some(&format!("Bearer {TOKEN}")),
            None,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": asked,
                    "capabilities": {},
                    "clientInfo": { "name": "kb-tests", "version": "0" },
                },
            }),
        )?;
        assert_eq!(
            response.status, 200,
            "initialize {asked}: {}",
            response.body
        );
        let body: Value = serde_json::from_str(&response.body)?;
        assert_eq!(
            body.pointer("/result/protocolVersion")
                .and_then(Value::as_str),
            Some(answered),
            "asked {asked}: {body}"
        );
    }
    Ok(())
}

#[test]
fn search_over_http_returns_identifiers_and_titles_and_no_bodies() -> TestResult {
    let (_dir, store, index) = populated()?;
    let server = served(&store, &index)?;
    let session = server.initialize()?;
    let (text, failed) =
        server.call(session.as_deref(), "search", &json!({ "query": "checker" }))?;
    assert!(!failed, "search failed: {text}");
    let payload: Value = serde_json::from_str(&text)?;
    assert_eq!(payload.get("count").and_then(Value::as_u64), Some(1));
    assert!(
        !text.contains("borrow checker"),
        "search returned the body: {text}"
    );
    Ok(())
}

/// The round trip, over the transport that will carry it in the deployment.
#[test]
fn a_record_written_over_http_is_found_and_read_over_http() -> TestResult {
    let (_dir, store, index) = populated()?;
    let server = served(&store, &index)?;
    let session = server.initialize()?;
    let (written, failed) = server.call(
        session.as_deref(),
        "put",
        &json!({ "document": "* Photosynthesis\n\nchloroplasts and the calvin cycle\n" }),
    )?;
    assert!(!failed, "put failed: {written}");
    let id = serde_json::from_str::<Value>(&written)?
        .get("id")
        .and_then(Value::as_str)
        .ok_or("put returned no id")?
        .to_owned();

    let (found, failed) = server.call(
        session.as_deref(),
        "search",
        &json!({ "query": "chloroplasts" }),
    )?;
    assert!(!failed, "search failed: {found}");
    assert!(found.contains(&id), "the record was not found: {found}");

    let (document, failed) = server.call(session.as_deref(), "get", &json!({ "id": &id }))?;
    assert!(!failed, "get failed: {document}");
    assert!(document.contains("calvin cycle"), "{document}");
    Ok(())
}

// ── authentication ──────────────────────────────────────────────────────

/// A surface that records whether it was reached, which is how "rejected
/// before any handler runs" is asserted rather than described.
struct CountingSurface {
    calls: Arc<AtomicUsize>,
}

impl ToolSurface for CountingSurface {
    fn specs(&self) -> Vec<ToolSpec> {
        Vec::new()
    }

    fn call(&self, _name: &str, _arguments: &Value) -> ToolOutcome {
        self.calls.fetch_add(1, Ordering::SeqCst);
        ToolOutcome::Text("reached".to_owned())
    }
}

/// Every rejected shape of the `Authorization` header, and the proof that none
/// of them reached a handler.
#[test]
fn requests_without_a_valid_token_are_rejected_before_any_handler_runs() -> TestResult {
    let calls = Arc::new(AtomicUsize::new(0));
    let server = Served::start(CountingSurface {
        calls: Arc::clone(&calls),
    })?;
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": "search", "arguments": { "query": "anything" } },
    });
    for header in [
        None,
        Some(String::new()),
        Some("Bearer".to_owned()),
        Some(format!("Bearer {TOKEN}x")),
        Some(format!("Bearer  {TOKEN}")),
        Some(TOKEN.to_owned()),
        Some(format!("Basic {TOKEN}")),
        Some(format!("Bearer {}", TOKEN.to_uppercase())),
    ] {
        let response = post(server.addr, header.as_deref(), None, &body)?;
        assert_eq!(
            response.status, 401,
            "header {header:?} was not rejected: {}",
            response.body
        );
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "a handler ran for an unauthenticated request"
    );
    Ok(())
}

/// The rejection says how to authenticate, because a 401 with no challenge
/// leaves a client guessing at the scheme.
#[test]
fn a_rejection_names_the_scheme_it_expects() -> TestResult {
    let (_dir, store, index) = populated()?;
    let server = served(&store, &index)?;
    let response = post(server.addr, None, None, &json!({ "jsonrpc": "2.0" }))?;
    assert_eq!(response.status, 401);
    assert_eq!(
        response
            .header("www-authenticate")
            .map(str::to_ascii_lowercase),
        Some("bearer".to_owned())
    );
    Ok(())
}

/// The scheme is matched case-insensitively, as RFC 7235 requires, while the
/// token is not.
#[test]
fn the_scheme_is_case_insensitive_and_the_token_is_not() -> TestResult {
    let (_dir, store, index) = populated()?;
    let server = served(&store, &index)?;
    let response = post(
        server.addr,
        Some(&format!("bEaReR {TOKEN}")),
        None,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "kb-tests", "version": "0" },
            },
        }),
    )?;
    assert_eq!(response.status, 200, "{}", response.body);
    Ok(())
}

// ── startup ─────────────────────────────────────────────────────────────

/// A port already in use is reported with the address in it, rather than as a
/// bare I/O failure the operator has to guess the cause of (ENG-004).
#[test]
fn a_port_already_in_use_is_reported_with_the_address() -> TestResult {
    let (_dir, store, index) = populated()?;
    let first = served(&store, &index)?;
    let taken = first.addr;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let error = runtime.block_on(async move {
        HttpServer::bind(HttpConfig::new(taken, TOKEN.to_owned()))
            .await
            .err()
    });
    let error = error.ok_or("binding an occupied port succeeded")?;
    assert!(
        error.to_string().contains(&taken.to_string()),
        "the error does not name the address: {error}"
    );
    Ok(())
}

/// Shutdown is a real path rather than process death: systemd will stop this
/// service, and a server with no graceful stop drops whatever it was serving.
#[test]
fn shutting_down_stops_the_listener() -> TestResult {
    let (_dir, store, index) = populated()?;
    let server = served(&store, &index)?;
    let addr = server.addr;
    server.initialize()?;
    drop(server);
    // The listener is closed once the accept loop has stopped; a connection
    // attempt either refuses or produces no response.
    let refused = TcpStream::connect(addr)
        .and_then(|mut stream| {
            stream.write_all(b"POST /mcp HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")?;
            let mut buffer = Vec::new();
            stream.read_to_end(&mut buffer)?;
            Ok(buffer.is_empty())
        })
        .unwrap_or(true);
    assert!(refused, "the listener is still answering after shutdown");
    Ok(())
}

/// A server that starts with no token configured would be an open door onto
/// the corpus, so it refuses to start and says which variable to set
/// (ENG-002, ENG-004).
#[test]
fn the_binary_refuses_to_serve_http_without_a_token() -> TestResult {
    for token in [None, Some("")] {
        let home = tempfile::tempdir()?;
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_kb-mcp"));
        command
            .env("HOME", home.path())
            .env("XDG_CONFIG_HOME", home.path())
            .env("KB_DB_PATH", home.path().join("kb.db"))
            .env_remove("KB_MCP_TOKEN")
            .args(["--http", "127.0.0.1:0"])
            .stdin(std::process::Stdio::null());
        if let Some(token) = token {
            command.env("KB_MCP_TOKEN", token);
        }
        let out = command.output()?;
        assert!(
            !out.status.success(),
            "the server started without a token ({token:?})"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("KB_MCP_TOKEN"),
            "the refusal does not name the variable: {stderr}"
        );
    }
    Ok(())
}

/// There is no default bind address: without `--http` the process is the
/// stdio server T020 built and listens on no socket at all, which is what
/// keeps the corpus off any interface nobody chose.
#[test]
fn the_binary_serves_stdio_when_no_address_is_given() -> TestResult {
    let home = tempfile::tempdir()?;
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_kb-mcp"))
        .env("HOME", home.path())
        .env("XDG_CONFIG_HOME", home.path())
        .env("KB_DB_PATH", home.path().join("kb.db"))
        .env("KB_MCP_TOKEN", TOKEN)
        .stdin(std::process::Stdio::null())
        .output()?;
    // Immediate end-of-input on stdin is how a stdio session ends, and the
    // server says so. A process that had gone looking for a socket instead
    // would still be running, or would have reported a bind.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("connection closed"),
        "the server did not run as a stdio session: {stderr}"
    );
    assert!(
        !stderr.contains("listening"),
        "the server bound a socket nobody asked for: {stderr}"
    );
    Ok(())
}

/// The deployment path, end to end: the real binary, bound to a real port,
/// answering an authenticated request and refusing an unauthenticated one,
/// then stopping when its supervisor asks it to.
///
/// It is stopped with `SIGTERM` rather than killed for the reason T020
/// recorded: a killed process never flushes its coverage profile, and the
/// module under test reads as unexecuted while every test passes. Under
/// systemd the same signal is what `systemctl stop` sends.
#[test]
fn the_binary_serves_http_and_stops_when_asked() -> TestResult {
    let (dir, _store, _index) = populated()?;
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_kb-mcp"))
        .env("HOME", dir.path())
        .env("XDG_CONFIG_HOME", dir.path())
        .env("KB_MCP_TOKEN", TOKEN)
        .args(["--http", "127.0.0.1:0"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    let mut stderr = std::io::BufReader::new(child.stderr.take().ok_or("no stderr on the server")?);
    let mut announced = String::new();
    std::io::BufRead::read_line(&mut stderr, &mut announced)?;
    let addr: SocketAddr = announced
        .trim()
        .rsplit_once("http://")
        .and_then(|(_, rest)| rest.strip_suffix("/mcp"))
        .ok_or_else(|| format!("the server did not announce an address: {announced:?}"))?
        .parse()?;

    let initialize = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "kb-tests", "version": "0" },
        },
    });
    let authenticated = post(addr, Some(&format!("Bearer {TOKEN}")), None, &initialize)?;
    assert_eq!(authenticated.status, 200, "{}", authenticated.body);
    let anonymous = post(addr, None, None, &initialize)?;
    assert_eq!(anonymous.status, 401, "{}", anonymous.body);

    let stopped = std::process::Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()?;
    assert!(stopped.success(), "could not signal the server");
    let status = child.wait()?;
    assert!(
        status.success(),
        "the server did not stop cleanly: {status}"
    );
    Ok(())
}

/// A bind that cannot happen is a failed start naming the address, not a
/// process that lingers having quietly served nothing (ENG-004).
#[test]
fn the_binary_reports_an_address_it_cannot_bind() -> TestResult {
    let (dir, store, index) = populated()?;
    let occupied = served(&store, &index)?;
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_kb-mcp"))
        .env("HOME", dir.path())
        .env("XDG_CONFIG_HOME", dir.path())
        .env("KB_MCP_TOKEN", TOKEN)
        .args(["--http", &occupied.addr.to_string()])
        .stdin(std::process::Stdio::null())
        .output()?;
    assert!(!out.status.success(), "the server started on a taken port");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(&occupied.addr.to_string()),
        "the failure does not name the address: {stderr}"
    );
    Ok(())
}

/// A surface whose handler starts a runtime of its own.
///
/// Not a contrivance: `kb search` embeds its query through a client that
/// blocks on its own runtime, which is what every handler that retrieves
/// anything now does (T030). The surface is synchronous by design — nothing in
/// [`kb::mcp`] is async — so a transport that runs it on the reactor thread
/// panics with "cannot start a runtime from within a runtime" and answers the
/// request never.
struct BlockingSurface;

impl ToolSurface for BlockingSurface {
    fn specs(&self) -> Vec<ToolSpec> {
        vec![ToolSpec {
            name: "search",
            description: "a handler that blocks",
            schema: json!({ "type": "object" }),
        }]
    }

    fn call(&self, _name: &str, _arguments: &Value) -> ToolOutcome {
        match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => {
                runtime.block_on(async {});
                ToolOutcome::Text("ran".to_owned())
            }
            Err(e) => ToolOutcome::Failed(e.to_string()),
        }
    }
}

/// A handler that blocks answers, rather than taking the connection down with
/// it.
#[test]
fn a_handler_that_blocks_is_not_run_on_the_reactor() -> TestResult {
    let server = Served::start(BlockingSurface)?;
    let session = server.initialize()?;
    let (text, failed) = server.call(session.as_deref(), "search", &json!({ "query": "any" }))?;
    assert!(!failed, "the blocking handler failed: {text}");
    assert_eq!(text, "ran");
    Ok(())
}
