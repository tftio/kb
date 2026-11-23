//! Concurrent writers against one store and one index (plan T003).
//!
//! On the server, `kb-mcp` write tools, the `kb queue drain` timer and a
//! scheduled `kb embed` write to the same store and the same index at the same
//! time. They are separate processes, which is the deployment shape, so the
//! tests run them as separate processes: one long-running `kb-mcp` that
//! dispatches tool calls concurrently, and short-lived `kb queue drain` and
//! `kb embed` runs that overlap it.
//!
//! Two kinds of fault are covered. The first is a conflict between writers on
//! an index that already exists. The second is the first open: the index does
//! not exist yet, and several processes, or several tool calls inside one
//! server, all try to create it at once. Every scenario starts from a store
//! and an index that have never been written, and is repeated over several
//! fresh worlds, because a race that fires one run in five is not shown by one
//! run.
//!
//! The assertions are about outcomes: every write is present in the store and
//! in the index, `kb fsck` finds no drift, every record has vectors and none
//! is stored twice, and no process or tool call failed.

#![allow(
    clippy::significant_drop_tightening,
    reason = "the mock endpoint must outlive every subprocess that calls it"
)]

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Barrier;

use serde_json::{Value, json};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const MODEL: &str = "fixture-model";

/// Fresh worlds per scenario.
const ROUNDS: usize = 6;

/// A store, an index and a queue that have never been written, and the
/// embeddings endpoint every process is configured to use.
struct World {
    dir: tempfile::TempDir,
    store: PathBuf,
    index: PathBuf,
    queue: PathBuf,
    url: String,
}

impl World {
    fn new(url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        Ok(Self {
            store: dir.path().join("store"),
            index: dir.path().join("index.db"),
            queue: dir.path().join("queue"),
            url: url.to_owned(),
            dir,
        })
    }

    /// Environment shared by every process, as a deployment sets it.
    fn configure(&self, command: &mut Command) {
        command
            .env("HOME", self.dir.path())
            .env("XDG_CONFIG_HOME", self.dir.path())
            .env("KB_STORE_PATH", &self.store)
            .env("KB_INDEX_PATH", &self.index)
            .env("KB_QUEUE_PATH", &self.queue)
            .env("KB_EMBEDDING_BASE_URL", &self.url)
            .env("KB_EMBEDDING_MODEL", MODEL)
            .env_remove("KB_EMBEDDING_API_KEY")
            .env_remove("KB_RERANK_BASE_URL")
            .env_remove("KB_MAIL_ROOT");
    }

    /// Run `kb` to completion; `Err` carries the process's own words.
    fn kb(&self, args: &[&str]) -> Result<String, String> {
        let mut command = Command::new(env!("CARGO_BIN_EXE_kb"));
        self.configure(&mut command);
        let output = command
            .args(args)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("kb {args:?} did not start: {e}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        if output.status.success() {
            Ok(stdout)
        } else {
            Err(format!(
                "kb {args:?} failed: {stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    /// Put submissions in the queue, as the server's ingest endpoint would.
    fn enqueue(&self, ids: &[String]) -> TestResult {
        let queue = kb::ingest::Queue::open(&self.queue)?;
        for id in ids {
            queue.enqueue(&kb::ingest::Submission {
                id: id.clone(),
                corpus: "kb".to_owned(),
                document: format!("* Capture {id}\n\nthe body of {id}, queued.\n"),
                provenance: None,
            })?;
        }
        Ok(())
    }
}

/// A mock embeddings endpoint answering every request with one vector.
fn endpoint() -> (mockito::ServerGuard, String) {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data":[{"embedding":[0.5,0.5]}]}"#)
        .create();
    let url = format!("{}/v1", server.url());
    (server, url)
}

/// A running `kb-mcp` over stdio that is sent many calls before any answer is
/// read, so the server holds them in flight together.
struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
}

impl Server {
    fn start(world: &World) -> Result<Self, Box<dyn std::error::Error>> {
        let mut command = Command::new(env!("CARGO_BIN_EXE_kb-mcp"));
        world.configure(&mut command);
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().ok_or("no stdin on the server")?;
        let stdout = BufReader::new(child.stdout.take().ok_or("no stdout on the server")?);
        let mut server = Self {
            child,
            stdin: Some(stdin),
            stdout,
            next_id: 1,
        };
        let initialize = server.send(
            "initialize",
            &json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "kb-tests", "version": "0" },
            }),
        )?;
        server.read_until(&[initialize])?;
        server.write_line(&json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }))?;
        Ok(server)
    }

    fn write_line(&mut self, message: &Value) -> TestResult {
        let stdin = self.stdin.as_mut().ok_or("the server is shut down")?;
        writeln!(stdin, "{message}")?;
        stdin.flush()?;
        Ok(())
    }

    fn send(&mut self, method: &str, params: &Value) -> Result<i64, Box<dyn std::error::Error>> {
        let id = self.next_id;
        self.next_id += 1;
        self.write_line(&json!({
            "jsonrpc": "2.0", "id": id, "method": method, "params": params,
        }))?;
        Ok(id)
    }

    /// Read lines until every id in `wanted` has been answered.
    fn read_until(
        &mut self,
        wanted: &[i64],
    ) -> Result<Vec<(i64, Value)>, Box<dyn std::error::Error>> {
        let mut answers = Vec::new();
        while answers.len() < wanted.len() {
            let mut line = String::new();
            if self.stdout.read_line(&mut line)? == 0 {
                return Err("the server closed before answering every call".into());
            }
            let parsed: Value = serde_json::from_str(line.trim())?;
            if let Some(id) = parsed.get("id").and_then(Value::as_i64)
                && wanted.contains(&id)
            {
                answers.push((id, parsed));
            }
        }
        Ok(answers)
    }

    /// Send every call without waiting, then collect every answer. A tool
    /// that failed, or a JSON-RPC error, is returned as `Err` text.
    fn call_all(
        &mut self,
        calls: &[(&str, Value)],
    ) -> Result<Vec<Result<Value, String>>, Box<dyn std::error::Error>> {
        let mut ids = Vec::new();
        for (name, arguments) in calls {
            ids.push(self.send(
                "tools/call",
                &json!({ "name": name, "arguments": arguments }),
            )?);
        }
        let mut answers = self.read_until(&ids)?;
        answers.sort_by_key(|(id, _)| *id);
        Ok(answers
            .into_iter()
            .map(|(_, response)| outcome(&response))
            .collect())
    }
}

fn outcome(response: &Value) -> Result<Value, String> {
    let Some(result) = response.get("result") else {
        return Err(format!("no result in {response}"));
    };
    let text = result
        .pointer("/content/0/text")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        return Err(text.to_owned());
    }
    serde_json::from_str(text).map_err(|e| format!("unreadable result {text}: {e}"))
}

impl Drop for Server {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.wait();
    }
}

fn put_call(id: &str) -> (&'static str, Value) {
    (
        "put",
        json!({ "id": id, "document": format!("* Note {id}\n\nthe body of {id}, over MCP.\n") }),
    )
}

/// Every call answered, none failed, and no embedding reported as failed.
fn assert_writes_succeeded(answers: &[Result<Value, String>]) -> Result<(), String> {
    for answer in answers {
        let body = answer
            .as_ref()
            .map_err(|reason| format!("a tool call failed: {reason}"))?;
        let failed = body
            .pointer("/embedding/records")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .any(|record| record.get("status").and_then(Value::as_str) == Some("failed"));
        if failed {
            return Err(format!("an embedding failed: {body}"));
        }
    }
    Ok(())
}

/// What the store, the index and `kb fsck` say once all writers have stopped.
fn assert_settled(world: &World, expected: &[String]) -> TestResult {
    let store = kb::store::GitBlobStore::open(&world.store)?;
    let index = kb::index::Index::open(&world.index)?;
    let indexed = index.records_in("kb")?;
    for id in expected {
        let name = kb::store::RefName::new(&format!("kb/{id}"))?;
        kb::store::BlobStore::read_ref(&store, &name)
            .map_err(|e| format!("{id} is not in the store: {e}"))?;
        assert!(indexed.contains(id), "{id} is stored but not indexed");
        assert!(
            !index.embedded_spans(id, MODEL)?.is_empty(),
            "{id} has no vectors"
        );
    }
    assert_eq!(
        indexed.len(),
        expected.len(),
        "the index holds extra records"
    );

    // Vectors are keyed by content, so a row cannot repeat; count them to see
    // that none was lost against the passages that need them, and that the
    // key really is unique.
    let conn = rusqlite::Connection::open(&world.index)?;
    let rows: i64 = conn.query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))?;
    let distinct: i64 = conn.query_row(
        "SELECT COUNT(*) FROM (SELECT DISTINCT stream_hash, span_start, span_len, model FROM embeddings)",
        [],
        |r| r.get(0),
    )?;
    assert_eq!(rows, distinct, "an embedding row is stored twice");

    let fsck = world
        .kb(&["fsck", "--no-legacy"])
        .map_err(|e| format!("fsck: {e}"))?;
    assert!(
        fsck.contains("the index agrees with the store"),
        "fsck reports drift: {fsck}"
    );
    // Nothing is left for a catch-up embed to do, so no vector was lost.
    let again = world.kb(&["embed", "--corpus", "kb"])?;
    assert!(
        again.contains(" 0 spans embedded") || again.contains(" 0 span embedded"),
        "a vector was missing after the writers stopped: {again}"
    );
    Ok(())
}

/// MCP writes, queue drains and `kb embed` all start at once on a store and an
/// index that have never been written, and keep going over each other.
#[test]
fn mcp_writes_queue_drains_and_embeds_overlap_safely() -> TestResult {
    let (_endpoint, url) = endpoint();
    for round in 0..ROUNDS {
        let world = World::new(&url)?;
        let mcp_ids: Vec<String> = (0..10).map(|n| format!("mcp-{round}-{n}")).collect();
        let queued: Vec<String> = (0..8).map(|n| format!("queued-{round}-{n}")).collect();
        world.enqueue(&queued)?;
        let barrier = Barrier::new(3);

        std::thread::scope(|scope| -> TestResult {
            let server = scope.spawn(|| -> Result<(), String> {
                let mut server = Server::start(&world).map_err(|e| e.to_string())?;
                barrier.wait();
                let puts: Vec<_> = mcp_ids.iter().map(|id| put_call(id)).collect();
                let answers = server.call_all(&puts).map_err(|e| e.to_string())?;
                assert_writes_succeeded(&answers)?;
                // A second burst edits records the first wrote, while the
                // other writers are still running.
                let edits: Vec<_> = mcp_ids
                    .iter()
                    .take(5)
                    .map(|id| ("tags_add", json!({ "id": id, "tags": ["burst"] })))
                    .chain(mcp_ids.iter().skip(5).map(|id| {
                        (
                            "update",
                            json!({ "id": id, "document": format!("* Note {id}\n\nrewritten.\n") }),
                        )
                    }))
                    .collect();
                let answers = server.call_all(&edits).map_err(|e| e.to_string())?;
                assert_writes_succeeded(&answers)?;
                Ok(())
            });
            let drains = scope.spawn(|| -> Result<(), String> {
                barrier.wait();
                for _ in 0..4 {
                    world.kb(&["queue", "drain"])?;
                }
                Ok(())
            });
            let embeds = scope.spawn(|| -> Result<(), String> {
                barrier.wait();
                for _ in 0..4 {
                    world.kb(&["embed", "--corpus", "kb"])?;
                }
                Ok(())
            });
            for (name, handle) in [("server", server), ("drain", drains), ("embed", embeds)] {
                handle
                    .join()
                    .map_err(|_| format!("the {name} writer panicked"))??;
            }
            Ok(())
        })?;

        let mut expected = mcp_ids;
        expected.extend(queued);
        assert_settled(&world, &expected)?;
    }
    Ok(())
}

/// Several processes open an index that does not exist, at the same moment.
/// Whichever gets there first creates it; none may fail or see it half made.
#[test]
fn processes_opening_a_missing_index_together_all_succeed() -> TestResult {
    let (_endpoint, url) = endpoint();
    for _ in 0..ROUNDS {
        let world = World::new(&url)?;
        let barrier = Barrier::new(8);
        std::thread::scope(|scope| -> TestResult {
            let mut handles = Vec::new();
            for n in 0..8 {
                let (world, barrier) = (&world, &barrier);
                handles.push(scope.spawn(move || -> Result<(), String> {
                    barrier.wait();
                    match n % 3 {
                        0 => world.kb(&["embed", "--corpus", "kb"]).map(drop),
                        1 => world.kb(&["queue", "drain"]).map(drop),
                        _ => world.kb(&["tags"]).map(drop),
                    }
                }));
            }
            for handle in handles {
                handle.join().map_err(|_| "a process thread panicked")??;
            }
            Ok(())
        })?;
        assert_settled(&world, &[])?;
    }
    Ok(())
}

/// Several tool calls reach one server back to back on a fresh index. The
/// server runs each on its own thread, so each opens the index itself.
#[test]
fn tool_calls_in_one_server_on_a_fresh_index_all_succeed() -> TestResult {
    let (_endpoint, url) = endpoint();
    for round in 0..ROUNDS {
        let world = World::new(&url)?;
        let mut server = Server::start(&world)?;
        let ids: Vec<String> = (0..10).map(|n| format!("fresh-{round}-{n}")).collect();
        let mut calls: Vec<_> = ids.iter().map(|id| put_call(id)).collect();
        calls.push(("recent", json!({})));
        calls.push(("tags", json!({})));
        let answers = server.call_all(&calls)?;
        assert_writes_succeeded(&answers)?;
        drop(server);
        assert_settled(&world, &ids)?;
    }
    Ok(())
}

/// The first writes of a brand-new store come from the server and the queue
/// worker at once: the store directory itself is created by both.
#[test]
fn a_new_store_created_by_the_server_and_the_worker_together_holds_both() -> TestResult {
    let (_endpoint, url) = endpoint();
    for round in 0..ROUNDS {
        let world = World::new(&url)?;
        let queued: Vec<String> = (0..4)
            .map(|n| format!("first-queued-{round}-{n}"))
            .collect();
        world.enqueue(&queued)?;
        let mcp_ids: Vec<String> = (0..4).map(|n| format!("first-mcp-{round}-{n}")).collect();
        let barrier = Barrier::new(2);
        std::thread::scope(|scope| -> TestResult {
            let server = scope.spawn(|| -> Result<(), String> {
                let mut server = Server::start(&world).map_err(|e| e.to_string())?;
                barrier.wait();
                let puts: Vec<_> = mcp_ids.iter().map(|id| put_call(id)).collect();
                let answers = server.call_all(&puts).map_err(|e| e.to_string())?;
                assert_writes_succeeded(&answers)?;
                Ok(())
            });
            let drain = scope.spawn(|| -> Result<(), String> {
                barrier.wait();
                world.kb(&["queue", "drain"]).map(drop)
            });
            server.join().map_err(|_| "the server writer panicked")??;
            drain.join().map_err(|_| "the drain writer panicked")??;
            Ok(())
        })?;
        let mut expected = mcp_ids;
        expected.extend(queued);
        assert_settled(&world, &expected)?;
    }
    Ok(())
}

/// Several writers write the same record at once. Whichever lands last is the
/// record; what must hold is that the index describes that one and not an
/// earlier one.
#[test]
fn concurrent_writes_to_one_record_leave_the_index_agreeing_with_the_store() -> TestResult {
    let (_endpoint, url) = endpoint();
    for round in 0..ROUNDS {
        let world = World::new(&url)?;
        let id = format!("shared-{round}");
        let mut server = Server::start(&world)?;
        let seeded = server.call_all(&[put_call(&id)])?;
        assert_writes_succeeded(&seeded)?;
        let edits: Vec<_> = (0..12)
            .map(|n| {
                (
                    "update",
                    json!({ "id": id, "document": format!("* Note {id}\n\nrevision {n}.\n") }),
                )
            })
            .collect();
        let answers = server.call_all(&edits)?;
        assert_writes_succeeded(&answers)?;
        drop(server);
        assert_settled(&world, std::slice::from_ref(&id))?;
    }
    Ok(())
}
