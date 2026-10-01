//! The ingest endpoint over real HTTP (T022).
//!
//! The hook's whole contract with the server is this endpoint: one bounded
//! post, an acknowledgement that means the bytes are durable, and nothing
//! else. It rides on the T021 transport behind the same bearer token, so the
//! authentication assertion is here too — an ingest endpoint that accepted
//! unauthenticated posts would be a way to write into the corpus that the MCP
//! surface's token does not guard.
//!
//! The client is written against a `TcpStream` for the reason the MCP tests
//! give: a status code that is right in Rust and wrong on the wire is wrong.

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};

use kb::ingest::{Ingest, Queue};
use kb::mcp_http::{HttpConfig, HttpServer};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const TOKEN: &str = "a-shared-secret-for-the-operator-network";

/// One HTTP response, as it arrived.
struct Response {
    status: u16,
    body: String,
}

/// Send one request and read the whole response.
fn send(
    addr: SocketAddr,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<&str>,
) -> Result<Response, Box<dyn std::error::Error>> {
    let mut request = format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n");
    if let Some(token) = token {
        write!(request, "Authorization: Bearer {token}\r\n")?;
    }
    if let Some(body) = body {
        request.push_str("Content-Type: application/json\r\n");
        write!(request, "Content-Length: {}\r\n", body.len())?;
    }
    request.push_str("\r\n");
    if let Some(body) = body {
        request.push_str(body);
    }

    let mut stream = TcpStream::connect(addr)?;
    stream.write_all(request.as_bytes())?;
    stream.flush()?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw)?;
    let (head, body) = raw.split_once("\r\n\r\n").ok_or("no header/body split")?;
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or("no status line")?
        .parse()?;
    Ok(Response {
        status,
        body: body.to_owned(),
    })
}

/// A running server with an ingest endpoint, and the queue behind it.
struct Served {
    addr: SocketAddr,
    shutdown: kb::mcp_http::Shutdown,
    thread: Option<std::thread::JoinHandle<()>>,
    queue: Queue,
    dir: tempfile::TempDir,
}

impl Served {
    fn start() -> Result<Self, Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let store = dir.path().join("store");
        let index = dir.path().join("index.db");
        kb::store::GitBlobStore::open_or_init(&store)?;
        drop(kb::index::Index::open_for_rebuild(&index)?);
        let queue = Queue::open(&dir.path().join("queue"))?;
        let ingest = Ingest::new(queue.clone(), index);

        let config = HttpConfig::new("127.0.0.1:0".parse::<SocketAddr>()?, TOKEN.to_owned());
        let (ready, started) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                drop(ready.send(Err("no runtime".to_owned())));
                return;
            };
            runtime.block_on(async move {
                let server = match HttpServer::bind(config).await {
                    Ok(server) => server.with_ingest(ingest),
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
                drop(server.serve(kb::mcp::KbTools::new()).await);
            });
        });
        let (addr, shutdown) = started.recv()??;
        Ok(Self {
            addr,
            shutdown,
            thread: Some(thread),
            queue,
            dir,
        })
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

fn submission(id: &str) -> String {
    serde_json::json!({
        "id": id,
        "corpus": "kb",
        "document": format!("* Session {id}\n\nthe body.\n"),
    })
    .to_string()
}

/// The acknowledgement the hook waits for, and what it means.
#[test]
fn a_posted_session_is_acknowledged_and_durable() -> TestResult {
    let server = Served::start()?;
    let response = send(
        server.addr,
        "POST",
        "/ingest",
        Some(TOKEN),
        Some(&submission("cc-1")),
    )?;
    assert_eq!(response.status, 202, "body: {}", response.body);
    assert!(
        response.body.contains("cc-1"),
        "the acknowledgement does not name what was accepted: {}",
        response.body
    );

    let pending = server.queue.pending()?;
    assert_eq!(pending.len(), 1, "the submission is not on disk");
    assert_eq!(
        pending.first().ok_or("nothing is pending")?.submission.id,
        "cc-1"
    );
    Ok(())
}

/// The endpoint is behind the same token as the tools. Without it nothing is
/// written, which is the property that matters: a rejected post must not be a
/// post that half happened.
#[test]
fn an_unauthenticated_post_is_refused_and_writes_nothing() -> TestResult {
    let server = Served::start()?;
    let response = send(
        server.addr,
        "POST",
        "/ingest",
        None,
        Some(&submission("cc-2")),
    )?;
    assert_eq!(response.status, 401);
    assert!(server.queue.pending()?.is_empty(), "it was enqueued anyway");

    let wrong = send(
        server.addr,
        "POST",
        "/ingest",
        Some("not-the-token"),
        Some(&submission("cc-2")),
    )?;
    assert_eq!(wrong.status, 401);
    assert!(server.queue.pending()?.is_empty());
    Ok(())
}

/// A malformed post is refused rather than queued, because a submission the
/// worker is guaranteed to dead-letter is one the client should be told about
/// while it can still do something.
#[test]
fn a_malformed_post_is_refused_rather_than_queued() -> TestResult {
    let server = Served::start()?;
    for body in ["not json at all", r#"{"corpus":"kb"}"#, r#"{"id":""}"#] {
        let response = send(server.addr, "POST", "/ingest", Some(TOKEN), Some(body))?;
        assert_eq!(response.status, 400, "accepted {body:?}: {}", response.body);
    }
    assert!(server.queue.pending()?.is_empty());
    Ok(())
}

/// What reconciliation asks: which sessions does the server already hold?
#[test]
fn the_server_reports_the_ids_it_holds() -> TestResult {
    let server = Served::start()?;
    let empty = send(
        server.addr,
        "GET",
        "/ingest/ids?corpus=kb",
        Some(TOKEN),
        None,
    )?;
    assert_eq!(empty.status, 200, "body: {}", empty.body);
    let ids: Vec<String> = serde_json::from_str(&empty.body)?;
    assert!(ids.is_empty(), "a fresh corpus holds nothing: {ids:?}");

    send(
        server.addr,
        "POST",
        "/ingest",
        Some(TOKEN),
        Some(&submission("cc-3")),
    )?;
    let pending = server.queue.pending()?;
    let store = kb::store::GitBlobStore::open_or_init(&server.dir.path().join("store"))?;
    let index = kb::index::Index::open(&server.dir.path().join("index.db"))?;
    kb::ingest::drain(&store, &index, &server.queue, None)?;
    drop(pending);

    let after = send(
        server.addr,
        "GET",
        "/ingest/ids?corpus=kb",
        Some(TOKEN),
        None,
    )?;
    let ids: Vec<String> = serde_json::from_str(&after.body)?;
    assert_eq!(ids, vec!["cc-3".to_owned()]);
    Ok(())
}

/// The ids endpoint is behind the token too: the set of session ids is itself
/// something an unauthenticated caller should not learn.
#[test]
fn the_id_listing_needs_the_token() -> TestResult {
    let server = Served::start()?;
    let response = send(server.addr, "GET", "/ingest/ids?corpus=kb", None, None)?;
    assert_eq!(response.status, 401);
    Ok(())
}

/// A server whose index is not readable reports the fault rather than
/// answering with an empty corpus, which reconciliation would read as "the
/// server holds nothing" and act on by re-posting everything.
#[test]
fn an_unreadable_index_is_a_fault_rather_than_an_empty_answer() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(&dir.path().join("queue"))?;
    let index = dir.path().join("not-a-database");
    std::fs::write(&index, b"certainly not sqlite")?;
    let ingest = Ingest::new(queue, index);
    assert!(
        ingest.ids("kb").is_err(),
        "a corrupt index answered as empty"
    );
    Ok(())
}

/// The routing table, without a socket.
/// The endpoint has to exist in the binary that is deployed, not only in a
/// server a test assembled in-process. Until T025 `kb-mcp` served the tools
/// alone, so every post from a hook would have been a 404 behind a valid
/// token. The queue is named on the command line — the same `--queue` /
/// `KB_QUEUE_PATH` the worker reads — so the server and `kb queue drain`
/// agree on where submissions go by construction.
#[test]
fn the_served_binary_accepts_posts_into_the_named_queue() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = dir.path().join("store");
    let index = dir.path().join("index.db");
    let queue_dir = dir.path().join("elsewhere").join("queue");
    kb::store::GitBlobStore::open_or_init(&store)?;
    drop(kb::index::Index::open_for_rebuild(&index)?);

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_kb-mcp"))
        .env("HOME", dir.path())
        .env("XDG_CONFIG_HOME", dir.path())
        .env("KB_MCP_TOKEN", TOKEN)
        .env_remove("KB_QUEUE_PATH")
        .arg("--http")
        .arg("127.0.0.1:0")
        .arg("--store")
        .arg(&store)
        .arg("--index")
        .arg(&index)
        .arg("--queue")
        .arg(&queue_dir)
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

    let response = send(
        addr,
        "POST",
        "/ingest",
        Some(TOKEN),
        Some(&submission("cc-9")),
    )?;
    assert_eq!(response.status, 202, "body: {}", response.body);
    let pending = Queue::open(&queue_dir)?.pending()?;
    assert_eq!(
        pending
            .iter()
            .map(|p| p.submission.id.as_str())
            .collect::<Vec<_>>(),
        vec!["cc-9"],
        "the submission did not land in the queue the server was given"
    );
    let listing = send(addr, "GET", "/ingest/ids?corpus=kb", Some(TOKEN), None)?;
    assert_eq!(listing.status, 200, "body: {}", listing.body);

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

#[test]
fn only_the_two_capture_paths_are_routed_to_the_endpoint() {
    use kb::mcp_http::{IngestRoute, ingest_route};
    assert_eq!(ingest_route("/ingest"), Some(IngestRoute::Accept));
    assert_eq!(ingest_route("/ingest/"), Some(IngestRoute::Accept));
    assert_eq!(ingest_route("/ingest/ids"), Some(IngestRoute::Ids));
    assert_eq!(ingest_route("/mcp"), None);
    assert_eq!(ingest_route("/ingest/ids/extra"), None);
}

/// The wrong method on a capture path is refused rather than falling through
/// to the tool surface, which would answer a GET on /ingest with a protocol
/// error and tell the caller nothing about what they got wrong.
#[test]
fn the_wrong_method_on_a_capture_path_says_so() -> TestResult {
    let server = Served::start()?;
    let got = send(server.addr, "GET", "/ingest", Some(TOKEN), None)?;
    assert_eq!(got.status, 405, "body: {}", got.body);
    let posted = send(server.addr, "POST", "/ingest/ids", Some(TOKEN), Some("{}"))?;
    assert_eq!(posted.status, 405, "body: {}", posted.body);
    Ok(())
}
