//! Export from the superseded database and import into the store.
//!
//! One way, by the operator's directive of 2026-08-14: `kb.db` is read and
//! never written, and becomes a frozen archive once its export is verified.
//! The verification is the task, so these tests are about fidelity rather
//! than about the transfer working at all.

use kb::index::{Index, Scope, rebuild};
use kb::migrate::{export_from_db, import_into_store};
use kb::store::{BlobStore, GitBlobStore, RefName};
use rusqlite::Connection;

mod common;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Build a small database in the superseded schema, standing in for the live
/// one: an authored note, a transcript, and a node whose body would trip a
/// naive exporter.
fn legacy_db(path: &std::path::Path) -> TestResult {
    let conn = common::legacy_db(path)?;
    for (id, body, tags) in [
        (
            "n-authored",
            "* Against Solipsism\n\nThe argument turns on whitespace.\n",
            vec!["philosophy"],
        ),
        (
            "n-transcript",
            "* Human [2026-08-14 09:00]\n\nWhat did we decide?\n\n\
             * Assistant [2026-08-14 09:01]\n\nTo pack but never prune.\n",
            vec!["conversation", "codex-session"],
        ),
        (
            "n-awkward",
            "* Unicode and quotes\n\n\u{e9}migr\u{e9} \u{2014} \"quoted\", 'single', \\backslash.\n",
            vec![],
        ),
    ] {
        let mut doc = kb::parser::parse_document(body)?;
        let placed: Vec<String> = tags.iter().map(|t| (*t).to_string()).collect();
        kb::storage::place_tags(&mut doc, &placed);
        common::legacy_insert(&conn, id, &doc)?;
    }
    Ok(())
}

/// The whole point of the export artifact: what comes out the far end is what
/// went in, byte for byte. Anything less and the archive is a lossy copy of a
/// corpus whose original is about to be frozen.
#[test]
fn node_content_round_trips_byte_identically() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    legacy_db(&db)?;
    let artifact = dir.path().join("export.jsonl");

    let exported = export_from_db(&db, &artifact)?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let imported = import_into_store(&artifact, &store)?;

    assert_eq!(exported.nodes, 3);
    assert_eq!(imported.records, 3);
    let conn = Connection::open(&db)?;
    for id in ["n-authored", "n-transcript", "n-awkward"] {
        let full = kb::storage::get_node_full(&conn, id)?.ok_or("node vanished")?;
        let rendered = kb::org_meta::render_with_metadata(
            id,
            &full.created_at,
            &full.updated_at,
            &full.document,
        );
        let header = kb::record::RecordHeader::parse(
            &store.get(&store.read_ref(&RefName::new(&format!("kb/{id}"))?)?)?,
        )?;
        let raw = store.get(&kb::store::BlobHash::new(header.raw.as_str())?)?;
        assert_eq!(
            String::from_utf8(raw)?,
            rendered,
            "{id} did not round-trip byte-identically"
        );
    }
    Ok(())
}

/// Every id must survive. A corpus that silently loses records during the one
/// migration it will ever get is the failure this plan exists to prevent.
#[test]
fn every_id_resolves_after_import() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    legacy_db(&db)?;
    let artifact = dir.path().join("export.jsonl");
    let exported = export_from_db(&db, &artifact)?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    import_into_store(&artifact, &store)?;

    let names: Vec<String> = store
        .list_refs("kb/")?
        .iter()
        .map(|n| n.as_str().to_owned())
        .collect();

    assert_eq!(names.len(), exported.nodes);
    for id in ["n-authored", "n-transcript", "n-awkward"] {
        assert!(names.contains(&format!("kb/{id}")), "{id} is missing");
    }
    Ok(())
}

/// A transcript's passages are its turns and a note's are its sections, so
/// what a record *is* has to survive the migration rather than being
/// flattened to one kind. Getting this wrong would be invisible until
/// retrieval quietly worsened.
#[test]
fn a_transcript_imports_as_a_transcript() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    legacy_db(&db)?;
    let artifact = dir.path().join("export.jsonl");
    export_from_db(&db, &artifact)?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    import_into_store(&artifact, &store)?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    let transcript = index.passages("n-transcript")?;
    let authored = index.passages("n-authored")?;

    assert_eq!(transcript.len(), 2, "turns were not preserved");
    assert!(transcript.iter().all(|p| p.level == "turn"));
    assert!(authored.iter().all(|p| p.level == "section"));
    Ok(())
}

/// Tags are a primary retrieval axis; losing them would degrade search in a
/// way no count would reveal.
#[test]
fn tags_survive_the_migration() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    legacy_db(&db)?;
    let artifact = dir.path().join("export.jsonl");
    export_from_db(&db, &artifact)?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    import_into_store(&artifact, &store)?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    let dump = index.dump()?;

    assert!(
        dump.contains("record_tags\tn-transcript\tconversation"),
        "{dump}"
    );
    assert!(
        dump.contains("record_tags\tn-authored\tphilosophy"),
        "{dump}"
    );
    Ok(())
}

/// The superseded database is read and never written. Its being 630MB and the
/// only copy of the corpus until the store holds it is why this is asserted
/// rather than assumed.
#[test]
fn the_source_database_is_not_written_during_export() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    legacy_db(&db)?;
    let before = std::fs::read(&db)?;
    let artifact = dir.path().join("export.jsonl");

    export_from_db(&db, &artifact)?;

    assert_eq!(
        std::fs::read(&db)?,
        before,
        "the export wrote to the source database"
    );
    Ok(())
}

/// Importing twice must not double the corpus. The import will be run more
/// than once during a migration that is being checked as it goes.
#[test]
fn importing_twice_is_idempotent() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    legacy_db(&db)?;
    let artifact = dir.path().join("export.jsonl");
    export_from_db(&db, &artifact)?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;

    import_into_store(&artifact, &store)?;
    import_into_store(&artifact, &store)?;

    assert_eq!(store.list_refs("kb/")?.len(), 3);
    Ok(())
}

/// Naming a database that is not there is the commonest operator error, and
/// must say so rather than produce an empty export that looks like a corpus
/// with nothing in it.
#[test]
fn exporting_from_a_missing_database_names_it() -> TestResult {
    let dir = tempfile::tempdir()?;

    let outcome = export_from_db(&dir.path().join("absent.db"), &dir.path().join("out.jsonl"));

    assert!(
        format!("{outcome:?}").contains("absent.db"),
        "outcome was: {outcome:?}"
    );
    assert!(outcome.is_err());
    Ok(())
}

/// An artifact that cannot be written is reported before anything is read,
/// rather than after an hour of work.
#[test]
fn exporting_to_an_unwritable_path_is_refused() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    legacy_db(&db)?;

    let outcome = export_from_db(&db, &dir.path().join("no-such-dir").join("out.jsonl"));

    assert!(outcome.is_err(), "outcome was: {outcome:?}");
    Ok(())
}

#[test]
fn importing_a_missing_artifact_names_it() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;

    let outcome = import_into_store(&dir.path().join("no-export.jsonl"), &store);

    assert!(
        format!("{outcome:?}").contains("no-export.jsonl"),
        "outcome was: {outcome:?}"
    );
    Ok(())
}

/// A corrupt artifact must stop the import rather than silently skip records.
/// Losing part of a corpus quietly is the failure this whole migration is
/// arranged to prevent.
#[test]
fn a_corrupt_artifact_line_stops_the_import() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    for (name, content) in [
        ("not-json.jsonl", "this is not json\n"),
        ("no-id.jsonl", "{\"org\": \"* A\\n\"}\n"),
        (
            "bad-time.jsonl",
            "{\"id\":\"n1\",\"org\":\"* A\\n\",\"created_at\":\"yesterday\",\
             \"updated_at\":\"2026-05-19T09:00:00Z\",\"tags\":[]}\n",
        ),
        (
            "bad-tag.jsonl",
            "{\"id\":\"n1\",\"org\":\"* A\\n\",\"created_at\":\"2026-05-19T09:00:00Z\",\
             \"updated_at\":\"2026-05-19T09:00:00Z\",\"tags\":[\"Not Kebab\"]}\n",
        ),
    ] {
        let path = dir.path().join(name);
        std::fs::write(&path, content)?;

        let outcome = import_into_store(&path, &store);

        assert!(outcome.is_err(), "{name} was accepted: {outcome:?}");
    }
    Ok(())
}

/// Blank lines are ignored rather than treated as records, so an artifact
/// that ends with a newline imports cleanly.
#[test]
fn blank_lines_in_an_artifact_are_skipped() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    legacy_db(&db)?;
    let artifact = dir.path().join("export.jsonl");
    export_from_db(&db, &artifact)?;
    let padded = dir.path().join("padded.jsonl");
    std::fs::write(
        &padded,
        format!("\n{}\n\n", std::fs::read_to_string(&artifact)?),
    )?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;

    let report = import_into_store(&padded, &store)?;

    assert_eq!(report.records, 3);
    assert_eq!(report.transcripts, 1);
    Ok(())
}
