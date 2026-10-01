//! MCP over streamable HTTP, with a bearer token in front of it.
//!
//! A second front end over the [`crate::mcp`] tool surface, not a second
//! implementation: the handlers, their schemas and their output are the ones
//! stdio already serves, and this module adds transport and authentication.
//! Like [`crate::mcp_stdio`] it is the seam where the protocol library lives
//! (`REPO_INVARIANTS.md` ENG-010).
//!
//! **Authentication happens before routing.** The token is checked on the way
//! in and a request that fails is answered `401` without the MCP service ever
//! seeing it, so no handler can run for an unauthenticated caller. The token
//! itself is injected from the environment and never read from a file in
//! either repository (ENG-002), and a server started without one refuses to
//! start rather than opening the corpus to whoever finds the port.
//!
//! **There is no default address.** Binding happens because somebody named an
//! address, which is what keeps the deployment's reachability a decision
//! rather than an accident; the Constraints put the corpus on the operator's
//! network and nowhere else.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt as _, Full};
use thiserror::Error;
use tower_service::Service as _;

use crate::mcp::ToolSurface;
use crate::mcp_stdio::McpServer;

/// The environment variable carrying the bearer token.
pub const TOKEN_VARIABLE: &str = "KB_MCP_TOKEN";

/// The environment variable carrying extra `Host` values to accept.
///
/// The transport rejects a `Host` it does not recognise, which defends a
/// locally running server against DNS rebinding. The bound address is always
/// accepted; a deployment reached by name — `kb.example`, or a VPN name —
/// needs that name listed here, and finding out otherwise means debugging a
/// rejection that looks like a client bug.
pub const ALLOWED_HOSTS_VARIABLE: &str = "KB_MCP_ALLOWED_HOSTS";

/// Why the HTTP server could not start or continue.
#[derive(Debug, Error)]
pub enum HttpError {
    /// No token was configured, so the server refused to start.
    #[error("no bearer token: set {TOKEN_VARIABLE} to a non-empty value")]
    MissingToken,
    /// The listener could not be established.
    #[error("cannot listen on {addr}: {source}")]
    Bind {
        /// The address that could not be bound.
        addr: SocketAddr,
        /// What the operating system reported.
        source: std::io::Error,
    },
    /// The listener stopped accepting connections.
    #[error("the listener failed: {0}")]
    Accept(std::io::Error),
}

/// What the server needs to know.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    addr: SocketAddr,
    token: String,
    allowed_hosts: Vec<String>,
}

impl HttpConfig {
    /// A configuration binding `addr` and accepting `token`.
    #[must_use]
    pub const fn new(addr: SocketAddr, token: String) -> Self {
        Self {
            addr,
            token,
            allowed_hosts: Vec::new(),
        }
    }

    /// Accept these `Host` values in addition to the bound address.
    #[must_use]
    pub fn with_allowed_hosts(
        mut self,
        hosts: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_hosts = hosts.into_iter().map(Into::into).collect();
        self
    }

    /// Read the token and any extra hosts from the environment.
    ///
    /// The address is not read here: it comes from the command line, because a
    /// server that could become network-reachable through an environment
    /// variable is one nobody has to decide to expose.
    ///
    /// # Errors
    ///
    /// [`HttpError::MissingToken`] if the token variable is unset or empty.
    pub fn from_env(addr: SocketAddr) -> Result<Self, HttpError> {
        #[allow(
            clippy::disallowed_methods,
            reason = "deployment configuration, read once at the process edge (REPO_INVARIANTS.md ENG-002, ENG-013)"
        )]
        let token = std::env::var(TOKEN_VARIABLE).unwrap_or_default();
        if token.trim().is_empty() {
            return Err(HttpError::MissingToken);
        }
        #[allow(
            clippy::disallowed_methods,
            reason = "deployment configuration, read once at the process edge (REPO_INVARIANTS.md ENG-013)"
        )]
        let hosts = std::env::var(ALLOWED_HOSTS_VARIABLE).unwrap_or_default();
        Ok(Self::new(addr, token).with_allowed_hosts(parse_allowed_hosts(&hosts)))
    }

    /// The `Host` values this configuration accepts beyond the bound address.
    #[must_use]
    pub fn allowed_hosts(&self) -> &[String] {
        &self.allowed_hosts
    }
}

/// Read a comma-separated host list, ignoring spacing and empty entries.
///
/// Pure, and separated from [`HttpConfig::from_env`] because the parsing is
/// what can be wrong: reading a variable cannot be tested in-process at all,
/// since setting one is `unsafe` in this edition and the crate denies it.
#[must_use]
pub fn parse_allowed_hosts(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|host| !host.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Whether a request may proceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Authorization {
    /// The request carried the configured token.
    Granted,
    /// It did not, and this is why — for the log, never for the response,
    /// which says only that a bearer token is required.
    Denied(DenialReason),
}

/// Why a request was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenialReason {
    /// No `Authorization` header at all.
    Absent,
    /// A header that is not a bearer credential.
    NotBearer,
    /// A bearer credential that is not the configured token.
    WrongToken,
}

/// Decide whether an `Authorization` header authenticates the caller.
///
/// Pure, and the whole of the authentication rule (ENG-008). The scheme is
/// matched case-insensitively as RFC 7235 requires; the token is compared
/// exactly, and in time that does not depend on how much of it matched, so a
/// caller cannot learn the secret one byte at a time.
#[must_use]
pub fn authorize(header: Option<&str>, token: &str) -> Authorization {
    let Some(header) = header else {
        return Authorization::Denied(DenialReason::Absent);
    };
    let Some((scheme, credential)) = header.split_once(' ') else {
        return Authorization::Denied(DenialReason::NotBearer);
    };
    if !scheme.eq_ignore_ascii_case("bearer") {
        return Authorization::Denied(DenialReason::NotBearer);
    }
    if constant_time_eq(credential.as_bytes(), token.as_bytes()) {
        Authorization::Granted
    } else {
        Authorization::Denied(DenialReason::WrongToken)
    }
}

/// Compare two byte strings without an early return.
///
/// Lengths differing is not itself secret — the token's length is a property
/// of the deployment rather than of any request — but which byte first differs
/// is, so the comparison always reads all of the shorter input.
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = u8::from(left.len() != right.len());
    for (a, b) in left.iter().zip(right.iter()) {
        difference |= a ^ b;
    }
    difference == 0
}

/// A handle that stops a running server.
///
/// Graceful shutdown is a real path rather than process death because this
/// runs under systemd (T023), which stops services by asking: a server with no
/// stop drops whatever it was in the middle of serving.
#[derive(Debug, Clone)]
pub struct Shutdown {
    signal: tokio::sync::watch::Sender<bool>,
}

impl Shutdown {
    /// Ask the server to stop accepting connections.
    ///
    /// `send_replace` rather than `send`, and the difference is not stylistic:
    /// `send` fails and leaves the value untouched when no receiver exists,
    /// so a stop arriving in the window between binding and the accept loop's
    /// first poll would be dropped and the server would run forever. A
    /// supervisor stopping a service immediately after starting it is exactly
    /// that window.
    pub fn stop(&self) {
        let _ = self.signal.send_replace(true);
    }
}

/// A bound, not yet serving, HTTP server.
pub struct HttpServer {
    listener: tokio::net::TcpListener,
    address: SocketAddr,
    config: HttpConfig,
    shutdown: tokio::sync::watch::Sender<bool>,
    /// Subscribed at bind time rather than when serving starts, so that the
    /// signal cannot be sent into a channel with no receivers.
    stopping: tokio::sync::watch::Receiver<bool>,
    /// The capture endpoint, when this server offers one. Optional because
    /// the same transport serves a read-only agent surface on a machine that
    /// ingests nothing, and an endpoint that writes should not exist there.
    ingest: Option<Arc<crate::ingest::Ingest>>,
}

impl HttpServer {
    /// Bind the configured address.
    ///
    /// Binding is separate from serving so that a caller — a test, or a
    /// supervisor wanting to report readiness — can learn the address before
    /// the first connection arrives, which matters when the address names port
    /// zero.
    ///
    /// # Errors
    ///
    /// [`HttpError::Bind`], naming the address, if it cannot be listened on.
    pub async fn bind(config: HttpConfig) -> Result<Self, HttpError> {
        let listener = tokio::net::TcpListener::bind(config.addr)
            .await
            .map_err(|source| HttpError::Bind {
                addr: config.addr,
                source,
            })?;
        let address = listener.local_addr().map_err(|source| HttpError::Bind {
            addr: config.addr,
            source,
        })?;
        let (shutdown, stopping) = tokio::sync::watch::channel(false);
        Ok(Self {
            listener,
            address,
            config,
            shutdown,
            stopping,
            ingest: None,
        })
    }

    /// Offer the capture endpoint on this server.
    ///
    /// Separate from [`HttpServer::bind`] so that binding stays the only thing
    /// that can fail for a reason the operator must act on, and so a server
    /// that serves only the agent surface says so by not calling this.
    #[must_use]
    pub fn with_ingest(mut self, ingest: crate::ingest::Ingest) -> Self {
        self.ingest = Some(Arc::new(ingest));
        self
    }

    /// Where it is actually listening, which is not always what was asked for:
    /// port zero means the operating system chose.
    #[must_use]
    pub const fn address(&self) -> SocketAddr {
        self.address
    }

    /// A handle that stops it.
    #[must_use]
    pub fn shutdown(&self) -> Shutdown {
        Shutdown {
            signal: self.shutdown.clone(),
        }
    }

    /// Serve `surface` until [`Shutdown::stop`] is called.
    ///
    /// # Errors
    ///
    /// [`HttpError::Accept`] if the listener itself fails. A connection that
    /// fails is logged and dropped: one client's broken pipe is not a reason
    /// to stop serving everybody else.
    pub async fn serve<S: ToolSurface + Send + Sync + 'static>(
        self,
        surface: S,
    ) -> Result<(), HttpError> {
        let surface = Arc::new(surface);
        let token = Arc::new(self.config.token.clone());
        let ingest = self.ingest.clone();
        let service = self.mcp_service(&surface);
        let mut stopping = self.stopping.clone();
        loop {
            let accepted = tokio::select! {
                () = wait_for_stop(&mut stopping) => return Ok(()),
                accepted = self.listener.accept() => accepted,
            };
            let (stream, peer) = match accepted {
                Ok(accepted) => accepted,
                Err(e) if is_transient(&e) => {
                    tracing::warn!("dropped an incoming connection: {e}");
                    continue;
                }
                Err(e) => return Err(HttpError::Accept(e)),
            };
            let service = service.clone();
            let token = Arc::clone(&token);
            let ingest = ingest.clone();
            tokio::spawn(async move {
                if let Err(e) = serve_connection(stream, service, token, ingest).await {
                    tracing::warn!("connection from {peer} ended: {e}");
                }
            });
        }
    }

    /// The MCP service this server puts behind the token.
    fn mcp_service<S: ToolSurface + Send + Sync + 'static>(
        &self,
        surface: &Arc<S>,
    ) -> McpService<S> {
        let mut allowed = vec![
            self.address.to_string(),
            self.address.ip().to_string(),
            "localhost".to_owned(),
        ];
        allowed.extend(self.config.allowed_hosts.iter().cloned());
        let config = rmcp::transport::streamable_http_server::StreamableHttpServerConfig::default()
            .with_allowed_hosts(allowed)
            // Stateless, with a plain JSON body per request. Every tool here
            // answers in one response and none of them sends the client
            // anything unprompted, so a session and an event stream would buy
            // nothing and cost two things that matter to a service under
            // systemd: server-side state that a restart invalidates, and a
            // long-lived connection per client. Sessions are removed from the
            // protocol as of 2026-07-28 in any case.
            .with_legacy_session_mode(false)
            .with_json_response(true);
        let surface = Arc::clone(surface);
        rmcp::transport::streamable_http_server::StreamableHttpService::new(
            move || Ok(McpServer::new(SharedSurface(Arc::clone(&surface)))),
            Arc::new(
                rmcp::transport::streamable_http_server::session::local::LocalSessionManager::default(),
            ),
            config,
        )
    }
}

/// The MCP transport service, with the tool surface it serves.
type McpService<S> = rmcp::transport::streamable_http_server::StreamableHttpService<
    McpServer<SharedSurface<S>>,
    rmcp::transport::streamable_http_server::session::local::LocalSessionManager,
>;

/// One tool surface, shared by every session.
///
/// The transport builds a handler per session and each needs its own value;
/// the surface itself is stateless — it opens the database per call — so they
/// share one rather than each opening their own.
pub struct SharedSurface<S>(Arc<S>);

impl<S: ToolSurface> ToolSurface for SharedSurface<S> {
    fn specs(&self) -> Vec<crate::mcp::ToolSpec> {
        self.0.specs()
    }

    fn call(&self, name: &str, arguments: &serde_json::Value) -> crate::mcp::ToolOutcome {
        self.0.call(name, arguments)
    }
}

/// Whether an accept error is worth continuing after.
///
/// A connection that died between the kernel accepting it and this process
/// taking it is the client's business; anything else means the listener is
/// gone, and continuing would spin.
fn is_transient(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::Interrupted
    )
}

/// Resolve once the shutdown handle has been asked to stop.
async fn wait_for_stop(stopping: &mut tokio::sync::watch::Receiver<bool>) {
    if *stopping.borrow() {
        return;
    }
    // The only error is a closed channel, which happens when the server itself
    // is dropped — at which point stopping is exactly right.
    drop(stopping.changed().await);
}

/// Serve one connection, checking the token before the MCP service sees a
/// request.
async fn serve_connection<S: ToolSurface + Send + Sync + 'static>(
    stream: tokio::net::TcpStream,
    service: McpService<S>,
    token: Arc<String>,
    ingest: Option<Arc<crate::ingest::Ingest>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let io = hyper_util::rt::TokioIo::new(stream);
    let guarded =
        hyper::service::service_fn(move |request: hyper::Request<hyper::body::Incoming>| {
            let mut service = service.clone();
            let token = Arc::clone(&token);
            let ingest = ingest.clone();
            async move {
                let offered = request
                    .headers()
                    .get(http::header::AUTHORIZATION)
                    .and_then(|value| value.to_str().ok());
                match authorize(offered, &token) {
                    // Routing happens after authentication, never before, so
                    // an unauthenticated caller cannot learn which paths exist
                    // by the difference between a 401 and a 404.
                    Authorization::Granted => match ingest_route(request.uri().path()) {
                        Some(route) => Ok(serve_ingest(route, request, ingest.as_deref()).await),
                        None => service.call(request).await,
                    },
                    Authorization::Denied(reason) => {
                        tracing::warn!("rejected an MCP request: {reason:?}");
                        Ok(unauthorized())
                    }
                }
            }
        });
    hyper::server::conn::http1::Builder::new()
        .serve_connection(io, guarded)
        .await?;
    Ok(())
}

/// The answer to a request that did not authenticate.
///
/// It names the scheme and nothing else. Saying which of the three ways it
/// failed would tell an unauthenticated caller whether they had guessed the
/// shape of the credential.
fn unauthorized() -> hyper::Response<http_body_util::combinators::BoxBody<Bytes, Infallible>> {
    let body = Full::new(Bytes::from_static(b"a bearer token is required\n")).boxed();
    let mut response = hyper::Response::new(body);
    *response.status_mut() = http::StatusCode::UNAUTHORIZED;
    response
        .headers_mut()
        .insert(http::header::WWW_AUTHENTICATE, WWW_AUTHENTICATE);
    response
        .headers_mut()
        .insert(http::header::CONTENT_TYPE, TEXT_PLAIN);
    response
}

/// The challenge a rejection carries.
const WWW_AUTHENTICATE: http::HeaderValue = http::HeaderValue::from_static("Bearer");

/// The content type a rejection carries.
const TEXT_PLAIN: http::HeaderValue = http::HeaderValue::from_static("text/plain; charset=utf-8");

#[cfg(test)]
mod tests {
    use super::{
        Authorization, DenialReason, HttpConfig, HttpError, HttpServer, TOKEN_VARIABLE, authorize,
        constant_time_eq, is_transient, parse_allowed_hosts,
    };

    const TOKEN: &str = "correct-horse-battery-staple";

    #[test]
    fn a_bearer_credential_matching_the_token_is_granted() {
        assert_eq!(
            authorize(Some(&format!("Bearer {TOKEN}")), TOKEN),
            Authorization::Granted
        );
    }

    /// RFC 7235 makes the scheme case-insensitive. The credential is not, and
    /// treating it as though it were would shrink the secret's alphabet.
    #[test]
    fn the_scheme_is_case_insensitive_and_the_credential_is_not() {
        assert_eq!(
            authorize(Some(&format!("bEaReR {TOKEN}")), TOKEN),
            Authorization::Granted
        );
        assert_eq!(
            authorize(Some(&format!("Bearer {}", TOKEN.to_uppercase())), TOKEN),
            Authorization::Denied(DenialReason::WrongToken)
        );
    }

    #[test]
    fn every_other_shape_of_header_is_denied() {
        for (header, expected) in [
            (None, DenialReason::Absent),
            (Some(String::new()), DenialReason::NotBearer),
            (Some("Bearer".to_owned()), DenialReason::NotBearer),
            (Some(format!("Basic {TOKEN}")), DenialReason::NotBearer),
            (Some(TOKEN.to_owned()), DenialReason::NotBearer),
            (Some("Bearer ".to_owned()), DenialReason::WrongToken),
            (Some(format!("Bearer  {TOKEN}")), DenialReason::WrongToken),
            (Some(format!("Bearer {TOKEN}x")), DenialReason::WrongToken),
        ] {
            assert_eq!(
                authorize(header.as_deref(), TOKEN),
                Authorization::Denied(expected),
                "header {header:?}"
            );
        }
    }

    #[test]
    fn the_comparison_reads_both_inputs_whatever_they_are() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abz"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(!constant_time_eq(b"", b"a"));
        assert!(constant_time_eq(b"", b""));
    }

    /// The host list is read leniently: an operator writing
    /// `kb.example, kb.example:8722` in a unit file should not have to think about
    /// spacing or a trailing comma.
    #[test]
    fn the_allowed_host_list_ignores_spacing_and_empty_entries() {
        assert_eq!(
            parse_allowed_hosts(" kb.example , kb.example:8722 ,, "),
            vec!["kb.example".to_owned(), "kb.example:8722".to_owned()]
        );
        assert!(parse_allowed_hosts("").is_empty());
    }

    /// The refusal names the variable to set, since a server that will not
    /// start is only actionable if it says what is missing (ENG-004).
    #[test]
    fn a_missing_token_is_refused_by_name() {
        assert!(
            HttpError::MissingToken.to_string().contains(TOKEN_VARIABLE),
            "{}",
            HttpError::MissingToken
        );
    }

    /// A stop that arrives before the server starts accepting must still be
    /// obeyed. Found by a test that hung: `watch::Sender::send` fails and
    /// leaves the value untouched when nothing is subscribed yet, so the
    /// signal was dropped and the accept loop waited forever — which is
    /// exactly what a supervisor stopping a service right after starting it
    /// would do.
    #[test]
    fn a_stop_arriving_before_the_accept_loop_is_still_obeyed() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a test runtime");
        let served = runtime.block_on(async {
            let addr = "127.0.0.1:0".parse().expect("a literal socket address");
            let server = HttpServer::bind(HttpConfig::new(addr, TOKEN.to_owned()))
                .await
                .expect("a bound server");
            server.shutdown().stop();
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                server.serve(crate::mcp::KbTools::new()),
            )
            .await
        });
        assert!(
            served.is_ok(),
            "the server ignored a stop sent before it began accepting"
        );
    }

    #[test]
    fn a_dead_incoming_connection_does_not_stop_the_listener() {
        for kind in [
            std::io::ErrorKind::ConnectionAborted,
            std::io::ErrorKind::ConnectionReset,
            std::io::ErrorKind::Interrupted,
        ] {
            assert!(is_transient(&std::io::Error::new(kind, "gone")));
        }
        assert!(!is_transient(&std::io::Error::other("the listener died")));
    }
}

/// The capture endpoint's two routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestRoute {
    /// Accept one submission.
    Accept,
    /// Report the ids the server holds.
    Ids,
}

/// Which capture route a path names, if any.
///
/// A free function rather than a match inside the handler so the routing table
/// is one readable thing and can be tested without a socket.
#[must_use]
pub fn ingest_route(path: &str) -> Option<IngestRoute> {
    match path.trim_end_matches('/') {
        "/ingest" => Some(IngestRoute::Accept),
        "/ingest/ids" => Some(IngestRoute::Ids),
        _ => None,
    }
}

/// The value of one query parameter, or `None`.
fn query_value<'a>(query: Option<&'a str>, key: &str) -> Option<&'a str> {
    query?.split('&').find_map(|pair| {
        let (name, value) = pair.split_once('=')?;
        (name == key).then_some(value)
    })
}

/// Serve one capture request.
///
/// A body is read to completion before anything is written, and the size is
/// bounded: this endpoint is reachable by anything on the operator's network
/// holding the token, and an unbounded read is a way to fill the queue disk
/// with one request.
async fn serve_ingest(
    route: IngestRoute,
    request: hyper::Request<hyper::body::Incoming>,
    ingest: Option<&crate::ingest::Ingest>,
) -> hyper::Response<http_body_util::combinators::BoxBody<Bytes, Infallible>> {
    let Some(ingest) = ingest else {
        return plain(http::StatusCode::NOT_FOUND, "this server does not ingest\n");
    };
    match route {
        IngestRoute::Ids => {
            if request.method() != http::Method::GET {
                return plain(http::StatusCode::METHOD_NOT_ALLOWED, "GET only\n");
            }
            let corpus = query_value(request.uri().query(), "corpus").unwrap_or("kb");
            match ingest.ids(corpus) {
                Ok(ids) => match serde_json::to_string(&ids) {
                    Ok(body) => json(http::StatusCode::OK, body),
                    Err(e) => plain(http::StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}\n")),
                },
                Err(e) => plain(http::StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}\n")),
            }
        }
        IngestRoute::Accept => {
            if request.method() != http::Method::POST {
                return plain(http::StatusCode::METHOD_NOT_ALLOWED, "POST only\n");
            }
            let collected = match http_body_util::BodyExt::collect(http_body_util::Limited::new(
                request.into_body(),
                MAX_SUBMISSION_BYTES,
            ))
            .await
            {
                Ok(collected) => collected.to_bytes(),
                Err(e) => {
                    return plain(
                        http::StatusCode::PAYLOAD_TOO_LARGE,
                        &format!("the submission was not read: {e}\n"),
                    );
                }
            };
            match ingest.accept(&collected) {
                // Accepted, not created: the bytes are durable and the record
                // does not exist yet. A 201 would say something this endpoint
                // deliberately does not promise.
                Ok(id) => json(
                    http::StatusCode::ACCEPTED,
                    serde_json::json!({ "accepted": id }).to_string(),
                ),
                Err(e) => plain(http::StatusCode::BAD_REQUEST, &format!("{e}\n")),
            }
        }
    }
}

/// The largest submission this endpoint will read.
///
/// Generous next to a rendered session and far short of a disk. The biggest
/// transcript in the operator's corpus renders to a few hundred kilobytes.
const MAX_SUBMISSION_BYTES: usize = 32 * 1024 * 1024;

/// A response with a JSON body.
fn json(
    status: http::StatusCode,
    body: String,
) -> hyper::Response<http_body_util::combinators::BoxBody<Bytes, Infallible>> {
    let mut response = hyper::Response::new(Full::new(Bytes::from(body)).boxed());
    *response.status_mut() = status;
    response.headers_mut().insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    response
}

/// A response with a plain-text body.
fn plain(
    status: http::StatusCode,
    body: &str,
) -> hyper::Response<http_body_util::combinators::BoxBody<Bytes, Infallible>> {
    let mut response = hyper::Response::new(Full::new(Bytes::from(body.to_owned())).boxed());
    *response.status_mut() = status;
    response.headers_mut().insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    response
}
