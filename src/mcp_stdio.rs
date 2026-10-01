//! MCP over stdio, adapting the protocol to [`crate::mcp::ToolSurface`].
//!
//! The whole of kb's dependency on an MCP library lives here
//! (`REPO_INVARIANTS.md` ENG-010). Handlers know nothing about JSON-RPC,
//! content blocks, protocol versions or capability negotiation; this module
//! translates in both directions and does nothing else, so replacing the
//! protocol library is a rewrite of one file rather than of the tool surface.
//!
//! **A tool that failed returns a result, not a protocol error.** MCP
//! distinguishes the two, and the distinction is about whose problem it is: a
//! protocol error says the server could not route the request, and clients
//! render it opaquely, so a caller who asked for a record that does not exist
//! would be told only that something went wrong internally. That is a worse
//! answer than "no node with id …", which is a fact the agent can act on.

use std::borrow::Cow;
use std::sync::Arc;

use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
    ListToolsResult, PaginatedRequestParams, ProtocolVersion, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{ErrorData as McpError, ServiceExt};

use crate::mcp::{ToolOutcome, ToolSurface};

/// What the server tells a client it is.
///
/// The instructions are the surface's own argument for its tiering, addressed
/// to the agent that has to obey it. An agent that does not know `search` is
/// cheap and `get` is expensive has no reason to prefer one.
const INSTRUCTIONS: &str = "A personal knowledge base of notes, session transcripts and \
     correspondence. The tools are tiered by cost: `search` returns identifiers and titles, \
     `context` returns what a record links to, and `get` returns one record's full text. \
     Search first, then fetch only what the results justify — a `get` for every hit spends \
     the budget you need for reasoning. `put` writes a record permanently.";

/// An MCP server over a tool surface.
pub struct McpServer<S> {
    surface: Arc<S>,
}

impl<S> McpServer<S> {
    /// A server offering `surface`.
    pub fn new(surface: S) -> Self {
        Self {
            surface: Arc::new(surface),
        }
    }
}

/// Translate a domain tool specification into the protocol's shape.
///
/// A schema that is not a JSON object is replaced by the empty object rather
/// than refused: the specifications are compile-time constants in this crate,
/// so a non-object here is a programming error that a running server should
/// survive as an unhelpful tool rather than as a failure to start.
fn as_tool(spec: &crate::mcp::ToolSpec) -> Tool {
    let schema = spec
        .schema
        .as_object()
        .cloned()
        .unwrap_or_else(serde_json::Map::new);
    Tool::new(spec.name, spec.description, Arc::new(schema))
}

/// The protocol revisions this server implements.
///
/// rmcp also knows 2026-07-28 and, left to its default, agrees to it whenever a
/// client asks — and a current Claude Code client does ask. That revision
/// changes what results must carry (discovery, cache hints), none of which this
/// handler produces, so the client accepted `initialize` and then rejected
/// `tools/list` ("Invalid result … ttlMs … cacheScope"). Negotiation is bounded to the revisions
/// the tests exercise; a newer request settles on 2025-11-25.
const SUPPORTED_PROTOCOL_VERSIONS: &[ProtocolVersion] = &[
    ProtocolVersion::V_2024_11_05,
    ProtocolVersion::V_2025_03_26,
    ProtocolVersion::V_2025_06_18,
    ProtocolVersion::V_2025_11_25,
];

impl<S: ToolSurface + Send + Sync + 'static> ServerHandler for McpServer<S> {
    fn supported_protocol_versions(&self) -> Cow<'static, [ProtocolVersion]> {
        Cow::Borrowed(SUPPORTED_PROTOCOL_VERSIONS)
    }

    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(INSTRUCTIONS)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(
            self.surface.specs().iter().map(as_tool).collect(),
        ))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let arguments = request
            .arguments
            .map_or_else(|| serde_json::json!({}), serde_json::Value::Object);
        // Off the reactor thread, always. The surface is synchronous by design
        // and its handlers do blocking work — SQLite, mu, and an embedding
        // client that drives its own runtime — so running one here would block
        // every other request on this thread and, in the embedding case, panic
        // with "cannot start a runtime from within a runtime". The transport
        // being async is the transport's business.
        let surface = Arc::clone(&self.surface);
        let name = request.name.to_string();
        let called = name.clone();
        let outcome = tokio::task::spawn_blocking(move || surface.call(&called, &arguments))
            .await
            .map_err(|e| {
                McpError::internal_error(format!("the {name} tool did not finish: {e}"), None)
            })?;
        let result = match outcome {
            ToolOutcome::Text(text) => CallToolResult::success(vec![ContentBlock::text(text)]),
            ToolOutcome::Failed(reason) => CallToolResult::error(vec![ContentBlock::text(reason)]),
        };
        Ok(CallToolResponse::from(result))
    }
}

/// Serve `surface` over stdin and stdout until the client disconnects.
///
/// # Errors
///
/// Returns the reason the transport could not be established or did not shut
/// down cleanly.
pub async fn serve<S: ToolSurface + Send + Sync + 'static>(
    surface: S,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = McpServer::new(surface)
        .serve(rmcp::transport::stdio())
        .await?;
    service.waiting().await?;
    Ok(())
}
