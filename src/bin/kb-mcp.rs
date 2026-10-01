#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)
)]
//! Binary entrypoint for kb's MCP server, over stdio or over HTTP.
//!
//! Thin by invariant (`REPO_INVARIANTS.md` CLI-001): it parses the address to
//! bind if there is one, resolves the database path, reads the bearer token
//! from the environment, and hands everything to the library. Everything an
//! agent can ask for is decided in `kb::mcp`, and how it is carried in
//! `kb::mcp_stdio` and `kb::mcp_http`.
//!
//! **stdout is the protocol** in the stdio case. A stray `println!` anywhere in
//! a handler would be framed as a JSON-RPC message and break the session, which
//! is why diagnostics here go to stderr — including in the HTTP case, so that
//! one binary behaves one way.
//!
//! **No address, no listener.** Without `--http` this is the stdio server and
//! binds nothing: reachability over a network is something somebody asks for.
//!
//! **The paths are the CLI's paths.** `--store`, `--index` and `--queue` take
//! the same environment variables `kb` reads, so the server and the worker
//! that drains its queue agree on where everything is by construction, and a
//! deployment — or a measurement against a snapshot — points both at one
//! place with one setting.

use clap::Parser;

/// kb's MCP server.
#[derive(Debug, Parser)]
#[command(name = "kb-mcp")]
struct Cli {
    /// Serve streamable HTTP on this address instead of stdio, e.g.
    /// `192.0.2.10:8722`. The bearer token is read from `KB_MCP_TOKEN` and the
    /// server refuses to start without one. There is no default: omitting this
    /// serves stdio and binds nothing.
    #[arg(long, value_name = "ADDR")]
    http: Option<std::net::SocketAddr>,
    /// Path to the blob store.
    #[arg(long, env = "KB_STORE_PATH", default_value_os_t = kb::store::default_store_path())]
    store: std::path::PathBuf,
    /// Path to the derived index.
    #[arg(long, env = "KB_INDEX_PATH", default_value_os_t = kb::index::default_index_path())]
    index: std::path::PathBuf,
    /// Path to the capture queue the HTTP server accepts into. Served only
    /// with `--http`: the ingest endpoint rides the same transport and token.
    #[arg(long, env = "KB_QUEUE_PATH", default_value_os_t = kb::ingest::default_queue_path())]
    queue: std::path::PathBuf,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("kb-mcp: could not start: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    // The store and the index at their configured locations. `KB_DB_PATH` is
    // not read: the superseded database serves no tool since T029.
    let surface = kb::mcp::KbTools::with_paths(&cli.store, &cli.index);
    match cli.http {
        Some(addr) => runtime.block_on(serve_http(surface, addr, &cli.queue, &cli.index)),
        None => match runtime.block_on(kb::mcp_stdio::serve(surface)) {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("kb-mcp: {e}");
                std::process::ExitCode::FAILURE
            }
        },
    }
}

/// Bind and serve HTTP, reporting the address actually bound.
///
/// The address is printed because `--http 127.0.0.1:0` is a legitimate way to
/// ask the operating system to choose, and because a service that says where
/// it is listening is one whose logs answer the first question anyone asks of
/// it.
///
/// The capture endpoint is served beside the tools (T022): `POST /ingest`
/// enqueues into `queue` and `GET /ingest/ids` reports what `index` holds,
/// behind the same token. A queue that cannot be opened is a refusal to start
/// rather than a server without capture, because a hook posting into a server
/// that silently has no queue would be acknowledged by a 404 nobody reads.
async fn serve_http(
    surface: kb::mcp::KbTools,
    addr: std::net::SocketAddr,
    queue: &std::path::Path,
    index: &std::path::Path,
) -> std::process::ExitCode {
    let config = match kb::mcp_http::HttpConfig::from_env(addr) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("kb-mcp: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let queue = match kb::ingest::Queue::open(queue) {
        Ok(queue) => queue,
        Err(e) => {
            eprintln!("kb-mcp: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let ingest = kb::ingest::Ingest::new(queue, index.to_path_buf());
    let server = match kb::mcp_http::HttpServer::bind(config).await {
        Ok(server) => server.with_ingest(ingest),
        Err(e) => {
            eprintln!("kb-mcp: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    eprintln!("kb-mcp: listening on http://{}/mcp", server.address());
    stop_on_signal(server.shutdown());
    match server.serve(surface).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kb-mcp: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Stop the server when the supervisor asks.
///
/// systemd stops a service with `SIGTERM` and a terminal with `SIGINT`; a
/// server that handles neither is killed mid-request, which is the difference
/// between a restart nobody notices and one that drops whatever was in flight.
fn stop_on_signal(shutdown: kb::mcp_http::Shutdown) {
    tokio::spawn(async move {
        wait_for_signal().await;
        eprintln!("kb-mcp: stopping");
        shutdown.stop();
    });
}

/// Resolve on the first stop signal this platform can deliver.
#[cfg(unix)]
async fn wait_for_signal() {
    let mut term = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(term) => term,
        Err(e) => {
            eprintln!("kb-mcp: cannot listen for SIGTERM: {e}");
            return;
        }
    };
    tokio::select! {
        _ = term.recv() => {}
        result = tokio::signal::ctrl_c() => {
            if let Err(e) = result {
                eprintln!("kb-mcp: cannot listen for interrupts: {e}");
            }
        }
    }
}

/// Resolve on the first stop signal this platform can deliver.
#[cfg(not(unix))]
async fn wait_for_signal() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        eprintln!("kb-mcp: cannot listen for interrupts: {e}");
    }
}
