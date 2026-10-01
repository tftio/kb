//! Projecting the store into a `SilverBullet` space (T024).
//!
//! The projection is derived state under the same discipline the index lives
//! under: nothing flows back, a page can always be thrown away, and a full
//! re-projection reproduces the space byte-for-byte. These tests hold that
//! discipline rather than the prettiness of the markdown — a projection that
//! renders beautifully and cannot be regenerated is the two-sources-of-truth
//! shape the storage invariants exist to prevent.

use std::path::Path;

use kb::project::{Directory, Projection, page_name, project, render};
use kb::record::{
    ArtifactKind, ContentHash, CorpusId, NormalizedStream, RecordHeader, RecordId, SourceRef,
};
use kb::store::{BlobHash, BlobStore, GitBlobStore, RefName};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A header for a note whose raw and stream hashes are placeholders. The
/// renderer reads them for the generation marker and never resolves them, so a
/// fixed digest keeps the rendered page stable across runs.
fn header(
    id: &str,
    kind: ArtifactKind,
    tags: &[&str],
) -> Result<RecordHeader, Box<dyn std::error::Error>> {
    Ok(RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new("kb")?,
        kind,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", id)?,
        tags: tags
            .iter()
            .map(|t| kb::record::Tag::new(t))
            .collect::<Result<Vec<_>, _>>()?,
        raw: ContentHash::new("aaaa1111")?,
        stream: ContentHash::new("bbbb2222")?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    })
}

/// Normalize org text the way ingest does, so what the renderer sees here is
/// what it sees in the store.
fn stream(kind: ArtifactKind, org: &str) -> Result<NormalizedStream, Box<dyn std::error::Error>> {
    Ok(kb::record::normalize(kind, org.as_bytes())?)
}

fn hash(hex: &str) -> Result<BlobHash, Box<dyn std::error::Error>> {
    Ok(BlobHash::new(hex)?)
}

/// The page name is the record's identity and nothing else. Titles collide —
/// 53 of the real corpus's 3,931 records share one, and a single transcript
/// title is carried by eleven — so a name derived from a title would rename
/// pages as unrelated records arrive.
#[test]
fn a_page_is_named_by_its_corpus_and_record_id() -> TestResult {
    let head = header(
        "b70049ea-001f-49fd-ba5e-4344fbde9d92",
        ArtifactKind::Note,
        &[],
    )?;
    assert_eq!(page_name(&head), "kb/b70049ea-001f-49fd-ba5e-4344fbde9d92");
    Ok(())
}

/// The marker is what makes staleness and hand edits detectable rather than
/// silent: a page that cannot say which record and which content it was made
/// from is indistinguishable from one somebody wrote by hand.
#[test]
fn a_page_carries_a_marker_naming_its_record_and_content_hash() -> TestResult {
    let head = header("alpha", ArtifactKind::Note, &["rust", "storage"])?;
    let body = stream(ArtifactKind::Note, "#+title: A note\n\nsome prose.\n")?;
    let page = render(&hash("c0ffee")?, &head, &body, &Directory::default());

    assert!(page.body.starts_with("---\n"), "{}", page.body);
    assert!(page.body.contains("kb_record: alpha\n"), "{}", page.body);
    assert!(
        page.body.contains("kb_record_hash: c0ffee\n"),
        "{}",
        page.body
    );
    assert!(page.body.contains("kb_corpus: kb\n"), "{}", page.body);
    assert!(page.body.contains("kb_kind: note\n"), "{}", page.body);
    assert!(page.body.contains("title: \"A note\"\n"), "{}", page.body);
    assert!(
        page.body.contains("tags: [rust, storage]\n"),
        "{}",
        page.body
    );
    Ok(())
}

/// Provenance is projected into frontmatter as `kb_*` keys, so the wiki
/// carries the same axis `kb get` shows. A record asserting none of it gains
/// none of these keys, rather than a run of empty ones.
#[test]
fn provenance_is_projected_into_frontmatter() -> TestResult {
    let mut head = header("alpha", ArtifactKind::Note, &[])?;
    head.provenance = kb::record::RawProvenance {
        project: Some("kb".to_owned()),
        project_source: Some("declared".to_owned()),
        remote: Some("github.com/tftio/kb".to_owned()),
        context: Some("personal".to_owned()),
        domains: vec!["clanker".to_owned()],
        harness: Some("claude-code".to_owned()),
        model: Some("claude-sonnet-5".to_owned()),
        session: Some("s-1".to_owned()),
        cwd: Some("/Users/op/Projects/kb/main".to_owned()),
    }
    .validate()?;
    let body = stream(ArtifactKind::Note, "#+title: A note\n\nsome prose.\n")?;

    let page = render(&hash("c0ffee")?, &head, &body, &Directory::default());

    assert!(page.body.contains("kb_project: kb\n"), "{}", page.body);
    assert!(
        page.body.contains("kb_project_source: declared\n"),
        "{}",
        page.body
    );
    assert!(
        page.body.contains("kb_remote: github.com/tftio/kb\n"),
        "{}",
        page.body
    );
    assert!(
        page.body.contains("kb_context: \"personal\"\n"),
        "{}",
        page.body
    );
    assert!(
        page.body.contains("kb_domains: [\"clanker\"]\n"),
        "{}",
        page.body
    );
    assert!(
        page.body.contains("kb_harness: \"claude-code\"\n"),
        "{}",
        page.body
    );
    assert!(
        page.body
            .contains("kb_cwd: \"/Users/op/Projects/kb/main\"\n"),
        "{}",
        page.body
    );
    Ok(())
}

/// A record with no provenance projects none of the `kb_*` provenance keys.
#[test]
fn a_record_with_no_provenance_projects_no_provenance_keys() -> TestResult {
    let head = header("alpha", ArtifactKind::Note, &[])?;
    let body = stream(ArtifactKind::Note, "#+title: A note\n\nsome prose.\n")?;

    let page = render(&hash("c0ffee")?, &head, &body, &Directory::default());

    assert!(!page.body.contains("kb_project"), "{}", page.body);
    assert!(!page.body.contains("kb_remote"), "{}", page.body);
    assert!(!page.body.contains("kb_context"), "{}", page.body);
    Ok(())
}

/// Determinism is the whole argument for disposability: if the same record can
/// render two ways, a re-projection is a diff and the space stops being
/// derived state.
#[test]
fn the_same_record_renders_byte_identically_every_time() -> TestResult {
    let head = header("alpha", ArtifactKind::Note, &["rust"])?;
    let body = stream(ArtifactKind::Note, "#+title: A note\n\n* One\n\ntext\n")?;
    let first = render(&hash("c0ffee")?, &head, &body, &Directory::default());
    let second = render(&hash("c0ffee")?, &head, &body, &Directory::default());
    assert_eq!(first.body, second.body);
    Ok(())
}

/// A title the document states is not repeated as a heading, because the
/// document did not write one; a document whose title *is* its first heading
/// keeps that heading and gains no second copy of it.
#[test]
fn a_stated_title_becomes_the_heading_and_a_heading_title_is_not_duplicated() -> TestResult {
    let head = header("alpha", ArtifactKind::Note, &[])?;
    let stated = render(
        &hash("c0ffee")?,
        &head,
        &stream(ArtifactKind::Note, "#+title: Stated\n\nprose.\n")?,
        &Directory::default(),
    );
    assert_eq!(
        stated.body.matches("# Stated").count(),
        1,
        "{}",
        stated.body
    );

    let head_only = render(
        &hash("c0ffee")?,
        &head,
        &stream(ArtifactKind::Note, "* Headed\n\nprose.\n")?,
        &Directory::default(),
    );
    assert_eq!(
        head_only.body.matches("# Headed").count(),
        1,
        "{}",
        head_only.body
    );
    Ok(())
}

/// Org structure has to survive the crossing, or the wiki shows something the
/// corpus does not say.
#[test]
fn org_structure_renders_as_markdown() -> TestResult {
    let head = header("alpha", ArtifactKind::Note, &[])?;
    let org = "\
* Top
** Nested

*bold* and /italic/ and =code=.

- first
- second

#+begin_src rust
let x = 1;
#+end_src

#+begin_quote
quoted
#+end_quote
";
    let page = render(
        &hash("c0ffee")?,
        &head,
        &stream(ArtifactKind::Note, org)?,
        &Directory::default(),
    );
    for wanted in [
        "# Top",
        "## Nested",
        "**bold**",
        "*italic*",
        "`code`",
        "- first",
        "```rust",
        "let x = 1;",
        "> quoted",
    ] {
        assert!(
            page.body.contains(wanted),
            "{wanted:?} missing from:\n{}",
            page.body
        );
    }
    Ok(())
}

/// An id link is the one link kind that cannot be ambiguous, so it is the one
/// that must always resolve when the target is in the projection.
#[test]
fn an_id_link_becomes_a_wikilink_to_the_targets_page() -> TestResult {
    let target = header("beta", ArtifactKind::Note, &[])?;
    let mut directory = Directory::default();
    directory.insert(&target, &stream(ArtifactKind::Note, "#+title: Beta\n")?);

    let head = header("alpha", ArtifactKind::Note, &[])?;
    let page = render(
        &hash("c0ffee")?,
        &head,
        &stream(ArtifactKind::Note, "see [[id:beta][the other note]].\n")?,
        &directory,
    );
    assert!(
        page.body.contains("[[kb/beta|the other note]]"),
        "{}",
        page.body
    );
    assert!(page.unresolved.is_empty(), "{:?}", page.unresolved);
    Ok(())
}

/// Bracket links in this corpus name a `#+name:` slug rather than an id, and
/// the projection is only useful if those keep working.
#[test]
fn a_name_link_resolves_through_the_slug_the_target_declares() -> TestResult {
    let target = header("beta", ArtifactKind::Note, &[])?;
    let mut directory = Directory::default();
    directory.insert(
        &target,
        &stream(
            ArtifactKind::Note,
            "#+name: user-biography\n#+title: Beta\n",
        )?,
    );

    let head = header("alpha", ArtifactKind::Note, &[])?;
    let page = render(
        &hash("c0ffee")?,
        &head,
        &stream(ArtifactKind::Note, "see [[user-biography]].\n")?,
        &directory,
    );
    assert!(
        page.body.contains("[[kb/beta|user-biography]]"),
        "{}",
        page.body
    );
    Ok(())
}

/// A wikilink to a page that does not exist is worse than no link: `SilverBullet`
/// renders it as an invitation to create the page, which would make the wiki a
/// write path into content the store has never seen.
#[test]
fn a_link_the_projection_cannot_resolve_renders_as_text_and_is_reported() -> TestResult {
    let head = header("alpha", ArtifactKind::Note, &[])?;
    let page = render(
        &hash("c0ffee")?,
        &head,
        &stream(ArtifactKind::Note, "see [[nothing-here]].\n")?,
        &Directory::default(),
    );
    assert!(!page.body.contains("[["), "{}", page.body);
    assert!(page.body.contains("nothing-here"), "{}", page.body);
    assert_eq!(page.unresolved, vec!["nothing-here".to_owned()]);
    Ok(())
}

/// A link out of the corpus is a link out of the corpus, and must not be
/// rewritten into a page reference.
#[test]
fn a_url_link_renders_as_an_ordinary_markdown_link() -> TestResult {
    let head = header("alpha", ArtifactKind::Note, &[])?;
    let page = render(
        &hash("c0ffee")?,
        &head,
        &stream(
            ArtifactKind::Note,
            "see [[https://example.com/x][the page]].\n",
        )?,
        &Directory::default(),
    );
    assert!(
        page.body.contains("[the page](https://example.com/x)"),
        "{}",
        page.body
    );
    assert!(page.unresolved.is_empty(), "{:?}", page.unresolved);
    Ok(())
}

// ── the shell: walking a store into a space ─────────────────────────────

/// Write a note into the store under `id`, as ingest would.
fn store_note(store: &GitBlobStore, id: &str, org: &str) -> TestResult {
    let normalized = kb::record::normalize(ArtifactKind::Note, org.as_bytes())?;
    let head = RecordHeader {
        raw: ContentHash::new(store.put(org.as_bytes())?.as_str())?,
        stream: ContentHash::new(store.put(normalized.as_bytes())?.as_str())?,
        ..header(id, ArtifactKind::Note, &[])?
    };
    let hash = store.put(&head.serialize())?;
    store.set_ref(&RefName::new(&format!("kb/{id}"))?, &hash)?;
    Ok(())
}

/// A store of two linked notes and an empty space directory beside it.
fn prepared() -> Result<(tempfile::TempDir, GitBlobStore), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_note(
        &store,
        "alpha",
        "#+title: Alpha\n#+name: alpha-note\n\npoints at [[id:beta][beta]].\n",
    )?;
    store_note(
        &store,
        "beta",
        "#+title: Beta\n\npoints back at [[alpha-note]].\n",
    )?;
    Ok((dir, store))
}

fn space(dir: &tempfile::TempDir) -> std::path::PathBuf {
    dir.path().join("space")
}

fn read(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(std::fs::read_to_string(path)?)
}

/// The acceptance property: projection is derived state, so projecting twice
/// produces the same space and the second run has nothing to write.
#[test]
fn projecting_twice_reproduces_the_space_byte_for_byte() -> TestResult {
    let (dir, store) = prepared()?;
    let first: Projection = project(&store, &space(&dir), false)?;
    let alpha = space(&dir).join("kb/alpha.md");
    let after_first = read(&alpha)?;

    let second = project(&store, &space(&dir), false)?;
    assert_eq!(read(&alpha)?, after_first);
    assert_eq!(first.written, 2);
    assert_eq!(second.written, 0);
    assert_eq!(second.unchanged, 2);
    Ok(())
}

/// Every wikilink the projection emits names a page the projection contains.
/// A link that does not is not a rendering defect; it is a claim about the
/// corpus the corpus does not support.
#[test]
fn every_wikilink_names_a_page_the_projection_wrote() -> TestResult {
    let (dir, store) = prepared()?;
    project(&store, &space(&dir), false)?;
    for name in ["alpha", "beta"] {
        let body = read(&space(&dir).join(format!("kb/{name}.md")))?;
        for link in wikilinks(&body) {
            let target = link.split('|').next().unwrap_or_default().to_owned();
            assert!(
                space(&dir).join(format!("{target}.md")).exists(),
                "{name} links to {target}, which the projection does not contain"
            );
        }
    }
    Ok(())
}

/// Every `[[…]]` target in a rendered page.
fn wikilinks(body: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = body;
    while let Some(open) = rest.find("[[") {
        let after = &rest[open + 2..];
        let Some(close) = after.find("]]") else { break };
        found.push(after[..close].to_owned());
        rest = &after[close + 2..];
    }
    found
}

/// A hand edit is overwritten and said out loud. Silence here would train the
/// operator to treat the wiki as editable, which is exactly the second write
/// path the projection is not allowed to become.
#[test]
fn a_hand_edited_page_is_reported_and_overwritten() -> TestResult {
    let (dir, store) = prepared()?;
    let projected = project(&store, &space(&dir), false)?;
    let alpha = space(&dir).join("kb/alpha.md");
    let generated = read(&alpha)?;
    std::fs::write(&alpha, format!("{generated}\n\nsomething a human typed\n"))?;

    let second = project(&store, &space(&dir), false)?;
    assert_eq!(second.hand_edited, vec!["kb/alpha".to_owned()]);
    assert_eq!(read(&alpha)?, generated);
    assert_eq!(projected.hand_edited.len(), 0);
    Ok(())
}

/// A record whose content moved on gets a refreshed page, and that is an
/// ordinary update rather than a hand edit: the marker distinguishes them.
#[test]
fn a_record_whose_content_changed_refreshes_its_page_without_alarm() -> TestResult {
    let (dir, store) = prepared()?;
    project(&store, &space(&dir), false)?;
    store_note(&store, "beta", "#+title: Beta\n\nrewritten entirely.\n")?;

    let second = project(&store, &space(&dir), false)?;
    assert_eq!(second.refreshed, 1);
    assert!(second.hand_edited.is_empty(), "{:?}", second.hand_edited);
    assert!(read(&space(&dir).join("kb/beta.md"))?.contains("rewritten entirely"));
    Ok(())
}

/// A page whose record left the store is reported, not deleted: deletion is
/// destructive and the operator's own pages share the directory.
#[test]
fn a_page_whose_record_is_gone_is_reported_and_left_alone() -> TestResult {
    let (dir, store) = prepared()?;
    project(&store, &space(&dir), false)?;
    std::fs::remove_file(dir.path().join("store/refs/kb/kb/beta"))?;

    let second = project(&store, &space(&dir), false)?;
    assert_eq!(second.orphaned, vec!["kb/beta".to_owned()]);
    assert!(space(&dir).join("kb/beta.md").exists());
    Ok(())
}

/// …and removed when the operator asks for it, which is the only way the space
/// converges on the store after a deletion.
#[test]
fn an_orphan_page_is_removed_under_prune() -> TestResult {
    let (dir, store) = prepared()?;
    project(&store, &space(&dir), false)?;
    std::fs::remove_file(dir.path().join("store/refs/kb/kb/beta"))?;

    let second = project(&store, &space(&dir), true)?;
    assert_eq!(second.pruned, 1);
    assert!(!space(&dir).join("kb/beta.md").exists());
    Ok(())
}

/// A page the projection did not write is not the projection's to touch, with
/// or without `--prune`. The space is the operator's directory; only pages
/// carrying a marker are claimed by kb.
#[test]
fn a_page_kb_never_wrote_is_left_alone_even_under_prune() -> TestResult {
    let (dir, store) = prepared()?;
    project(&store, &space(&dir), false)?;
    let theirs = space(&dir).join("kb/handwritten.md");
    std::fs::write(&theirs, "# mine\n")?;

    let second = project(&store, &space(&dir), true)?;
    assert_eq!(read(&theirs)?, "# mine\n");
    assert_eq!(second.pruned, 0);
    assert!(second.orphaned.is_empty(), "{:?}", second.orphaned);
    Ok(())
}

/// One-directional, asserted rather than asserted-about: a hand-edited space
/// leaves the store's refs and objects exactly as they were.
#[test]
fn projection_never_writes_the_store() -> TestResult {
    let (dir, store) = prepared()?;
    project(&store, &space(&dir), false)?;
    std::fs::write(space(&dir).join("kb/alpha.md"), "# rewritten by hand\n")?;
    let before = store_state(&store)?;

    project(&store, &space(&dir), true)?;
    assert_eq!(store_state(&store)?, before);
    Ok(())
}

/// Every ref in the store and the hash it points at.
fn store_state(store: &GitBlobStore) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let mut state = Vec::new();
    for name in store.list_refs("")? {
        let hash = store.read_ref(&name)?;
        state.push((name.as_str().to_owned(), hash.as_str().to_owned()));
    }
    Ok(state)
}

/// The projection reports its cost, because the claim that it is disposable is
/// only true while re-running it is cheap enough that people do.
#[test]
fn a_projection_reports_what_it_cost() -> TestResult {
    let (dir, store) = prepared()?;
    let done = project(&store, &space(&dir), false)?;
    assert_eq!(done.records, 2);
    assert!(done.elapsed.as_nanos() > 0);
    Ok(())
}
