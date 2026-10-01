//! Projecting the store into a `SilverBullet` space.
//!
//! The wiki is derived state, held to the same discipline as the index: it can
//! be deleted and regenerated, nothing in it is authoritative, and nothing
//! flows back. What keeps that true is not a prohibition but a marker — every
//! page names the record and the content hash it was made from, so a page that
//! has drifted from its source, or that somebody edited by hand, is a fact the
//! next run can state rather than a difference nobody can see.
//!
//! **Rendering is a pure function of the record** ([`render`]), with the
//! directory walk and the writes confined to [`project`] (ENG-008). That split
//! is what makes byte-identical re-projection testable without a filesystem,
//! and it is the property the whole disposability claim rests on: if the same
//! record could render two ways, a re-projection would be a diff and the space
//! would have become a second source of truth.
//!
//! **Pages are named by record id**, not by title. Titles collide — 53 of the
//! real corpus's 3,931 records share one and a single transcript title is
//! carried by eleven — so a title-derived name would rename existing pages as
//! unrelated records arrive, and every link into them with it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::record::{ArtifactKind, NormalizedStream, RecordHeader};
use crate::store::{BlobHash, BlobStore};
use tftio_org::ast::{Block, Checkbox, Document, Inline, ListItem, ListType, TableCell};

/// Why a projection could not be completed.
#[derive(Debug, Error)]
pub enum ProjectError {
    /// The store could not be enumerated or read.
    #[error("reading {record} from the store: {source}")]
    Store {
        /// Which ref was being read.
        record: String,
        /// What the store reported.
        source: crate::store::StoreError,
    },
    /// A record blob will not parse as a record header.
    #[error("parsing {record}: {source}")]
    Record {
        /// Which record could not be parsed.
        record: String,
        /// What the parse reported.
        source: crate::record::RecordError,
    },
    /// The space directory could not be read or written.
    #[error("projecting into {path}: {source}")]
    Space {
        /// Which path was involved.
        path: String,
        /// What the filesystem reported.
        source: std::io::Error,
    },
}

/// One rendered page.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Page {
    /// The page's name in the space, without the `.md` suffix.
    pub name: String,
    /// The page's whole content, frontmatter included.
    pub body: String,
    /// Link targets naming nothing the projection contains. Reported rather
    /// than emitted: `SilverBullet` renders a wikilink to a missing page as an
    /// invitation to create it, which would make the wiki a write path into
    /// content the store has never seen.
    pub unresolved: Vec<String>,
}

/// Where every record will land, so links can be rewritten to pages that exist.
///
/// Both indexes are needed because this corpus writes links two ways: org
/// `[[id:…]]` references, and bracket references naming a document's `#+name:`
/// slug. Neither can be resolved from the record being rendered — only from
/// the whole corpus — which is why projection is two passes over the store.
#[derive(Debug, Clone, Default)]
pub struct Directory {
    by_id: BTreeMap<String, String>,
    by_slug: BTreeMap<String, String>,
}

impl Directory {
    /// Record where `header` will be projected, and under which slug.
    pub fn insert(&mut self, header: &RecordHeader, stream: &NormalizedStream) {
        let page = page_name(header);
        if let Some(slug) = name_slug(header.kind, stream) {
            self.by_slug.insert(slug, page.clone());
        }
        self.by_id.insert(header.id.as_str().to_owned(), page);
    }

    /// The page a record id projects to, if the projection contains it.
    #[must_use]
    pub fn page_for_id(&self, id: &str) -> Option<&str> {
        self.by_id.get(id).map(String::as_str)
    }

    /// The page a `#+name:` slug projects to, if the projection contains it.
    #[must_use]
    pub fn page_for_slug(&self, slug: &str) -> Option<&str> {
        self.by_slug.get(slug).map(String::as_str)
    }
}

/// What a projection run did.
///
/// The three write outcomes are kept apart rather than summed: a refreshed
/// page is an ordinary consequence of the corpus moving on, while a
/// hand-edited one means somebody treated the wiki as editable, and reporting
/// forty of the first as forty of the second would train its reader to ignore
/// both.
#[derive(Debug, Clone, Default)]
pub struct Projection {
    /// Records walked.
    pub records: usize,
    /// Pages written that did not exist before.
    pub written: usize,
    /// Pages rewritten because their record's content moved.
    pub refreshed: usize,
    /// Pages already holding exactly what this run would write.
    pub unchanged: usize,
    /// Pages whose content was not what kb last wrote for that same record,
    /// overwritten.
    pub hand_edited: Vec<String>,
    /// Pages kb wrote whose record the store no longer holds.
    pub orphaned: Vec<String>,
    /// Orphan pages deleted, which happens only when asked.
    pub pruned: usize,
    /// Link targets naming nothing the projection contains, as
    /// `<page> -> <target>`.
    pub unresolved: Vec<String>,
    /// How long the run took.
    pub elapsed: Duration,
}

/// Where the space lives when nothing says otherwise.
///
/// `$HOME/.local/share/kb/space`, beside the store and the index. Derived
/// state like the index, and disposable in the same way: deleting it costs a
/// re-projection.
#[must_use]
pub fn default_space_path() -> PathBuf {
    dirs::home_dir().map_or_else(
        || PathBuf::from("space"),
        |home| home.join(".local/share/kb/space"),
    )
}

/// The page name a record projects to: `<corpus>/<record id>`.
#[must_use]
pub fn page_name(header: &RecordHeader) -> String {
    format!("{}/{}", header.corpus.as_str(), header.id.as_str())
}

/// Render one record to its page.
///
/// Pure and total: a stream that will not parse as org is rendered verbatim
/// rather than reported, because a projection that fails on one record leaves
/// the operator with no wiki at all, and the record itself remains
/// authoritative either way.
#[must_use]
pub fn render(
    record: &BlobHash,
    header: &RecordHeader,
    stream: &NormalizedStream,
    directory: &Directory,
) -> Page {
    let name = page_name(header);
    let document = parsed(header.kind, stream);
    let title = title_of(document.as_ref(), header);
    let mut unresolved = Vec::new();
    let body = document.as_ref().map_or_else(
        || verbatim(stream),
        |doc| blocks(&doc.blocks, directory, &mut unresolved),
    );

    let mut out = frontmatter(record, header, &title.text);
    out.push('\n');
    // A title the document only *states* has no heading of its own, so the
    // page gains one; a title that came from a heading already has it, and
    // emitting a second copy would show every note twice-titled.
    if title.stated {
        let _ = writeln!(out, "# {}\n", title.text);
    }
    out.push_str(&body);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    Page {
        name,
        body: out,
        unresolved,
    }
}

/// Walk the store and write every record's page into `space`.
///
/// Two passes, because a link can only be rewritten against the whole corpus:
/// the first builds the [`Directory`], the second renders. Orphan pages —
/// those kb wrote whose record the store no longer holds — are reported, and
/// deleted only when `prune` says so. Deletion is destructive and the space is
/// the operator's own directory, so it is not something a scheduled job should
/// do by default.
///
/// # Errors
///
/// [`ProjectError`] if the store cannot be read or the space cannot be written.
pub fn project(
    store: &impl BlobStore,
    space: &Path,
    prune: bool,
) -> Result<Projection, ProjectError> {
    let started = Instant::now();
    let mut done = Projection::default();

    let names = store.list_refs("").map_err(|source| ProjectError::Store {
        record: "<enumerating>".to_owned(),
        source,
    })?;
    let mut records = Vec::with_capacity(names.len());
    let mut directory = Directory::default();
    for name in &names {
        let label = name.as_str().to_owned();
        let address = store.read_ref(name).map_err(|source| ProjectError::Store {
            record: label.clone(),
            source,
        })?;
        let blob = store.get(&address).map_err(|source| ProjectError::Store {
            record: label.clone(),
            source,
        })?;
        let header = RecordHeader::parse(&blob).map_err(|source| ProjectError::Record {
            record: label.clone(),
            source,
        })?;
        directory.insert(&header, &stream_of(store, &header)?);
        records.push((address, header));
    }

    let mut projected = BTreeSet::new();
    for (address, header) in &records {
        let page = render(address, header, &stream_of(store, header)?, &directory);
        done.records += 1;
        done.unresolved.extend(
            page.unresolved
                .iter()
                .map(|target| format!("{} -> {target}", page.name)),
        );
        projected.insert(page.name.clone());
        write_page(space, &page, address, &mut done)?;
    }

    for page in kb_pages(space)? {
        if projected.contains(&page.name) {
            continue;
        }
        done.orphaned.push(page.name.clone());
        if prune {
            std::fs::remove_file(&page.path).map_err(|source| ProjectError::Space {
                path: page.path.display().to_string(),
                source,
            })?;
            done.pruned += 1;
        }
    }
    done.orphaned.sort();
    done.elapsed = started.elapsed();
    Ok(done)
}

/// Read the normalized stream a record's header addresses.
fn stream_of(
    store: &impl BlobStore,
    header: &RecordHeader,
) -> Result<NormalizedStream, ProjectError> {
    let bytes = store
        .get(&BlobHash::from(&header.stream))
        .map_err(|source| ProjectError::Store {
            record: header.id.as_str().to_owned(),
            source,
        })?;
    Ok(NormalizedStream::from_bytes(bytes))
}

/// Write one page, classifying what was there before it.
///
/// The marker decides the classification. A page holding kb's last output for
/// this exact record content, but not matching it, was edited by somebody; a
/// page holding kb's output for *different* content is simply stale; a page
/// with no marker at this name was not written by kb at all, which is the same
/// hazard as a hand edit and is reported the same way.
fn write_page(
    space: &Path,
    page: &Page,
    record: &BlobHash,
    done: &mut Projection,
) -> Result<(), ProjectError> {
    let path = page_path(space, &page.name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| ProjectError::Space {
            path: parent.display().to_string(),
            source,
        })?;
    }
    match std::fs::read_to_string(&path) {
        Ok(existing) if existing == page.body => {
            done.unchanged += 1;
            return Ok(());
        }
        Ok(existing) => {
            if marker_hash(&existing).is_none_or(|hash| hash == record.as_str()) {
                done.hand_edited.push(page.name.clone());
            } else {
                done.refreshed += 1;
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => done.written += 1,
        Err(source) => {
            return Err(ProjectError::Space {
                path: path.display().to_string(),
                source,
            });
        }
    }
    std::fs::write(&path, &page.body).map_err(|source| ProjectError::Space {
        path: path.display().to_string(),
        source,
    })
}

/// Where a page name lands under the space directory.
fn page_path(space: &Path, name: &str) -> PathBuf {
    space.join(format!("{name}.md"))
}

/// A page kb wrote, found on disk.
struct SpacePage {
    /// Page name, without the `.md` suffix.
    name: String,
    /// Where it is.
    path: PathBuf,
}

/// Every markdown file under `space` carrying kb's generation marker.
///
/// Only marker-bearing pages are kb's to reason about. The space is the
/// operator's directory and may hold their own notes; a file kb did not write
/// is never reported as an orphan and never deleted.
fn kb_pages(space: &Path) -> Result<Vec<SpacePage>, ProjectError> {
    let mut found = Vec::new();
    collect_pages(space, space, &mut found)?;
    found.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(found)
}

/// Recursive half of [`kb_pages`].
fn collect_pages(root: &Path, dir: &Path, found: &mut Vec<SpacePage>) -> Result<(), ProjectError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        // A space that does not exist yet holds no pages. Everything else is
        // reported: an unreadable directory would otherwise be indistinguishable
        // from an empty one, and the projection would quietly stop finding
        // orphans (ENG-004).
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(ProjectError::Space {
                path: dir.display().to_string(),
                source,
            });
        }
    };
    for entry in entries {
        let entry = entry.map_err(|source| ProjectError::Space {
            path: dir.display().to_string(),
            source,
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_pages(root, &path, found)?;
            continue;
        }
        if path.extension().is_none_or(|ext| ext != "md") {
            continue;
        }
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        if marker_hash(&body).is_none() {
            continue;
        }
        if let Some(name) = page_name_of(root, &path) {
            found.push(SpacePage { name, path });
        }
    }
    Ok(())
}

/// The page name a file on disk carries, relative to the space root.
fn page_name_of(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    let text = relative.to_str()?;
    Some(text.trim_end_matches(".md").replace('\\', "/"))
}

/// The content hash a page's generation marker names.
fn marker_hash(body: &str) -> Option<&str> {
    body.lines()
        .take_while(|line| *line != "---" || body.starts_with("---"))
        .find_map(|line| line.strip_prefix(MARKER_HASH))
        .map(str::trim)
}

/// The frontmatter key naming the record blob a page was rendered from.
const MARKER_HASH: &str = "kb_record_hash: ";

/// A record's title, and whether the document stated it rather than heading
/// with it.
struct PageTitle {
    text: String,
    stated: bool,
}

/// The record's title: what `#+title:` says, else its first heading, else the
/// record id, which is never pretty and never absent.
fn title_of(document: Option<&Document>, header: &RecordHeader) -> PageTitle {
    let Some(document) = document else {
        return PageTitle {
            text: header.id.as_str().to_owned(),
            stated: true,
        };
    };
    for block in &document.blocks {
        if let Block::Keyword { name, value } = block
            && name.eq_ignore_ascii_case("title")
            && !value.trim().is_empty()
        {
            return PageTitle {
                text: value.trim().to_owned(),
                stated: true,
            };
        }
    }
    for block in &document.blocks {
        if let Block::Heading { title, .. } = block {
            return PageTitle {
                text: title.as_str().to_owned(),
                stated: false,
            };
        }
    }
    PageTitle {
        text: header.id.as_str().to_owned(),
        stated: true,
    }
}

/// The `#+name:` slug a document declares, which is what bracket links in this
/// corpus point at.
fn name_slug(kind: ArtifactKind, stream: &NormalizedStream) -> Option<String> {
    let document = parsed(kind, stream)?;
    document.blocks.iter().find_map(|block| match block {
        Block::Keyword { name, value } if name.eq_ignore_ascii_case("name") => {
            let slug = value.trim();
            (!slug.is_empty()).then(|| slug.to_owned())
        }
        _ => None,
    })
}

/// Parse a stream's payload as org, when its kind implies org at all.
fn parsed(kind: ArtifactKind, stream: &NormalizedStream) -> Option<Document> {
    match kind {
        ArtifactKind::Note | ArtifactKind::SessionTranscript => {
            let text = std::str::from_utf8(stream.payload()).ok()?;
            crate::parser::parse_document(text).ok()
        }
        // Mail is referenced rather than copied, so no mail record reaches the
        // store to be projected; were one to, its stream is readable text
        // rather than org and is rendered as such.
        ArtifactKind::MailMessage | ArtifactKind::MailThread => None,
    }
}

/// A stream that is not org, shown as it is.
fn verbatim(stream: &NormalizedStream) -> String {
    String::from_utf8_lossy(stream.payload()).into_owned()
}

/// The page's YAML frontmatter, which is both what `SilverBullet` reads and the
/// generation marker the next run compares against.
fn frontmatter(record: &BlobHash, header: &RecordHeader, title: &str) -> String {
    let mut out = String::from("---\n");
    let _ = writeln!(out, "title: \"{}\"", yaml_escape(title));
    if !header.tags.is_empty() {
        let tags = header
            .tags
            .iter()
            .map(|tag| tag.as_str().to_owned())
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "tags: [{tags}]");
    }
    let _ = writeln!(out, "kb_record: {}", header.id.as_str());
    let _ = writeln!(out, "kb_corpus: {}", header.corpus.as_str());
    let _ = writeln!(out, "kb_kind: {}", header.kind.as_str());
    let _ = writeln!(out, "{MARKER_HASH}{record}");
    let _ = writeln!(out, "kb_stream_hash: {}", header.stream.as_str());
    let _ = writeln!(out, "kb_created: {}", header.created.to_rfc3339());
    let _ = writeln!(out, "kb_updated: {}", header.updated.to_rfc3339());
    write_provenance(&mut out, &header.provenance);
    out.push_str("---\n");
    out
}

/// Where the work behind this page happened, projected into frontmatter as
/// `kb_*` keys so the wiki carries the same axis `kb get` shows.
fn write_provenance(out: &mut String, provenance: &crate::record::Provenance) {
    if let Some(project) = &provenance.project {
        let _ = writeln!(out, "kb_project: {}", project.as_str());
    }
    if let Some(source) = provenance.project_source {
        let _ = writeln!(out, "kb_project_source: {}", source.label());
    }
    if let Some(remote) = &provenance.remote {
        let _ = writeln!(out, "kb_remote: {}", remote.as_str());
    }
    if let Some(context) = &provenance.context {
        let _ = writeln!(out, "kb_context: \"{}\"", yaml_escape(context));
    }
    if !provenance.domains.is_empty() {
        let domains = provenance
            .domains
            .iter()
            .map(|d| format!("\"{}\"", yaml_escape(d)))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "kb_domains: [{domains}]");
    }
    if let Some(harness) = &provenance.harness {
        let _ = writeln!(out, "kb_harness: \"{}\"", yaml_escape(harness));
    }
    if let Some(model) = &provenance.model {
        let _ = writeln!(out, "kb_model: \"{}\"", yaml_escape(model));
    }
    if let Some(session) = &provenance.session {
        let _ = writeln!(out, "kb_session: \"{}\"", yaml_escape(session));
    }
    if let Some(cwd) = &provenance.cwd {
        let _ = writeln!(out, "kb_cwd: \"{}\"", yaml_escape(cwd));
    }
}

/// Escape a YAML double-quoted scalar.
fn yaml_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Render a run of blocks, separated by one blank line.
///
/// Org's explicit `BlankLine` blocks are dropped rather than carried across:
/// markdown's spacing is generated here, so honouring both would make a page's
/// shape depend on how its source happened to be spaced.
fn blocks(blocks: &[Block], directory: &Directory, unresolved: &mut Vec<String>) -> String {
    let mut out = String::new();
    for block in blocks {
        let rendered = block_text(block, directory, unresolved);
        if rendered.trim().is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(rendered.trim_end_matches('\n'));
        out.push('\n');
    }
    out
}

/// Render one block.
///
/// Dispatched exhaustively (ENG-009). Org bookkeeping — drawers, planning
/// lines, keyword lines, comments — carries no wiki content and is dropped:
/// the record remains authoritative for it, and a comment rendered as markdown
/// would become a heading.
fn block_text(block: &Block, directory: &Directory, unresolved: &mut Vec<String>) -> String {
    match block {
        Block::Heading {
            level,
            title,
            children,
            ..
        } => {
            let depth = "#".repeat(usize::from(*level).clamp(1, 6));
            let mut out = format!("{depth} {}\n", title.as_str());
            let nested = blocks(children, directory, unresolved);
            if !nested.is_empty() {
                out.push('\n');
                out.push_str(&nested);
            }
            out
        }
        Block::Paragraph { inlines } => {
            let mut out = inlines_text(inlines, directory, unresolved);
            out.push('\n');
            out
        }
        Block::SrcBlock { language, content } => fenced(language, content),
        Block::ExampleBlock { content } => fenced("", content),
        Block::QuoteBlock { children } => blocks(children, directory, unresolved)
            .lines()
            .map(|line| {
                if line.is_empty() {
                    ">\n".to_owned()
                } else {
                    format!("> {line}\n")
                }
            })
            .collect(),
        Block::List { list_type, items } => list_text(list_type, items, directory, unresolved),
        Block::Table { rows } => table_text(rows, directory, unresolved),
        Block::HorizontalRule => "---\n".to_owned(),
        Block::PropertyDrawer { .. }
        | Block::LogbookDrawer { .. }
        | Block::Planning { .. }
        | Block::Comment { .. }
        | Block::Keyword { .. }
        | Block::BlankLine => String::new(),
    }
}

/// A fenced code block, with the fence long enough to contain its content.
fn fenced(language: &str, content: &str) -> String {
    let fence = "`".repeat(longest_backtick_run(content).max(2).saturating_add(1));
    let body = content.trim_end_matches('\n');
    format!("{fence}{language}\n{body}\n{fence}\n")
}

/// The longest run of backticks in `text`, so a fence can be made longer.
fn longest_backtick_run(text: &str) -> usize {
    let mut longest = 0;
    let mut run = 0;
    for ch in text.chars() {
        if ch == '`' {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    longest
}

/// Render a list, indenting nested content by two spaces.
fn list_text(
    list_type: &ListType,
    items: &[ListItem],
    directory: &Directory,
    unresolved: &mut Vec<String>,
) -> String {
    let mut out = String::new();
    for (offset, item) in items.iter().enumerate() {
        let marker = match list_type {
            ListType::Ordered(start) => {
                format!("{}.", start.saturating_add(offset as u64))
            }
            ListType::Unordered => "-".to_owned(),
        };
        let checkbox = match item.checkbox {
            Checkbox::NoCheckbox => "",
            Checkbox::Unchecked => "[ ] ",
            Checkbox::Checked => "[x] ",
        };
        let content = blocks(&item.content, directory, unresolved);
        let mut lines = content.lines();
        let first = lines.next().unwrap_or_default();
        let _ = writeln!(out, "{marker} {checkbox}{first}");
        for line in lines {
            if line.is_empty() {
                out.push('\n');
            } else {
                let _ = writeln!(out, "  {line}");
            }
        }
    }
    out
}

/// Render a table, treating the first row as its header.
///
/// Org's rule rows reach the AST as ordinary cells holding `---+---`, so they
/// are recognised and dropped: emitting one would render as a literal row of
/// hyphens *and* leave the table without the separator markdown requires. The
/// separator is generated after the first row instead, which is the only place
/// markdown allows it.
fn table_text(
    rows: &[Vec<TableCell>],
    directory: &Directory,
    unresolved: &mut Vec<String>,
) -> String {
    let mut out = String::new();
    let mut emitted = 0usize;
    for row in rows {
        let cells: Vec<String> = row
            .iter()
            .map(|cell| {
                inlines_text(&cell.inlines, directory, unresolved)
                    .trim()
                    .to_owned()
            })
            .collect();
        if is_rule_row(&cells) {
            continue;
        }
        let _ = writeln!(out, "| {} |", cells.join(" | "));
        emitted = emitted.saturating_add(1);
        if emitted == 1 {
            let rule = cells.iter().map(|_| "---").collect::<Vec<_>>().join(" | ");
            let _ = writeln!(out, "| {rule} |");
        }
    }
    out
}

/// Whether a row is org's horizontal rule rather than content.
fn is_rule_row(cells: &[String]) -> bool {
    !cells.is_empty()
        && cells
            .iter()
            .all(|cell| !cell.is_empty() && cell.chars().all(|c| c == '-' || c == '+'))
}

/// Render inline content.
fn inlines_text(inlines: &[Inline], directory: &Directory, unresolved: &mut Vec<String>) -> String {
    let mut out = String::new();
    for inline in inlines {
        match inline {
            Inline::Plain(text) => out.push_str(text),
            Inline::Bold(children) => {
                let _ = write!(out, "**{}**", inlines_text(children, directory, unresolved));
            }
            Inline::Italic(children) => {
                let _ = write!(out, "*{}*", inlines_text(children, directory, unresolved));
            }
            Inline::Strikethrough(children) => {
                let _ = write!(out, "~~{}~~", inlines_text(children, directory, unresolved));
            }
            Inline::InlineCode(text) | Inline::Verbatim(text) => {
                let _ = write!(out, "`{text}`");
            }
            Inline::LineBreak => out.push('\n'),
            Inline::Link {
                target,
                description,
            } => out.push_str(&link_text(
                target,
                description.as_deref(),
                directory,
                unresolved,
            )),
        }
    }
    out
}

/// Rewrite one link.
///
/// Three kinds, and the distinction is the point. An id or slug naming a
/// record the projection contains becomes a wikilink; a URL stays a URL; and
/// anything else becomes plain text and is reported, because a wikilink to a
/// page that does not exist reads in `SilverBullet` as an offer to create it.
fn link_text(
    target: &str,
    description: Option<&str>,
    directory: &Directory,
    unresolved: &mut Vec<String>,
) -> String {
    let shown = description.unwrap_or(target);
    if let Some(id) = target.strip_prefix("id:") {
        return resolved(directory.page_for_id(id), shown, target, unresolved);
    }
    // A `#custom-id` reference points inside the document that wrote it. It is
    // not a corpus reference and reporting it as an unresolvable one would put
    // a quarter of this report — 172 of the corpus's 2,921 name links — on
    // targets that were never records.
    if target.contains("://") || target.starts_with("mailto:") || target.starts_with('#') {
        return format!("[{shown}]({target})");
    }
    resolved(directory.page_for_slug(target), shown, target, unresolved)
}

/// A wikilink when the projection contains the page, and reported plain text
/// when it does not.
fn resolved(page: Option<&str>, shown: &str, target: &str, unresolved: &mut Vec<String>) -> String {
    page.map_or_else(
        || {
            unresolved.push(target.to_owned());
            shown.to_owned()
        },
        |page| wikilink(page, shown),
    )
}

/// A wikilink to `page`, shown as `alias`.
///
/// An alias carrying the wikilink's own delimiters is dropped rather than
/// escaped: a mangled alias is a broken link, whereas the bare page name is
/// always a working one.
fn wikilink(page: &str, alias: &str) -> String {
    let clean = alias.replace(['|', '[', ']'], "");
    let clean = clean.trim();
    if clean.is_empty() || clean == page {
        format!("[[{page}]]")
    } else {
        format!("[[{page}|{clean}]]")
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Directory, MARKER_HASH, ProjectError, longest_backtick_run, marker_hash, page_name_of,
        project, render, wikilink, yaml_escape,
    };
    use crate::record::{
        ArtifactKind, ContentHash, CorpusId, NormalizedStream, RecordHeader, RecordId, SourceRef,
    };
    use crate::store::{BlobHash, BlobStore, GitBlobStore, RefName};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn header(kind: ArtifactKind) -> Result<RecordHeader, Box<dyn std::error::Error>> {
        Ok(RecordHeader {
            id: RecordId::new("alpha")?,
            corpus: CorpusId::new("kb")?,
            kind,
            created: "2026-05-19T09:00:00Z".parse()?,
            updated: "2026-08-14T10:30:00Z".parse()?,
            source: SourceRef::new("node-id", "alpha")?,
            tags: vec![],
            raw: ContentHash::new("aaaa1111")?,
            stream: ContentHash::new("bbbb2222")?,
            normalizer: crate::record::NORMALIZER_VERSION,
            provenance: crate::record::Provenance::default(),
        })
    }

    /// A record whose stream is not org — which the store cannot hold today,
    /// mail being referenced rather than copied — is shown rather than lost.
    #[test]
    fn a_stream_that_is_not_org_is_rendered_verbatim() -> TestResult {
        let stream = NormalizedStream::from_bytes(b"kb-stream/3\nfrom: someone\n\nbody\n".to_vec());
        let page = render(
            &BlobHash::new("c0ffee")?,
            &header(ArtifactKind::MailMessage)?,
            &stream,
            &Directory::default(),
        );
        assert!(page.body.contains("from: someone"), "{}", page.body);
        // No org document means no title to state, so the record's id names it.
        assert!(page.body.contains("title: \"alpha\""), "{}", page.body);
        Ok(())
    }

    /// A note with no title and no heading still gets a page name and a title:
    /// its id, which is never absent.
    #[test]
    fn a_titleless_note_is_titled_by_its_id() -> TestResult {
        let stream = crate::record::normalize(ArtifactKind::Note, b"just prose.\n")?;
        let page = render(
            &BlobHash::new("c0ffee")?,
            &header(ArtifactKind::Note)?,
            &stream,
            &Directory::default(),
        );
        assert!(page.body.contains("title: \"alpha\""), "{}", page.body);
        Ok(())
    }

    /// Tables, ordered lists and checkboxes cross over, and a table's
    /// separator row is generated because org's does not survive parsing.
    #[test]
    fn tables_and_lists_render() -> TestResult {
        let org = "\
| a | b |
|---+---|
| 1 | 2 |

1. first
2. second

- [ ] undone
- [X] done
";
        let stream = crate::record::normalize(ArtifactKind::Note, org.as_bytes())?;
        let page = render(
            &BlobHash::new("c0ffee")?,
            &header(ArtifactKind::Note)?,
            &stream,
            &Directory::default(),
        );
        for wanted in [
            "| a | b |",
            "| --- | --- |",
            "1. first",
            "- [ ] undone",
            "- [x] done",
        ] {
            assert!(
                page.body.contains(wanted),
                "{wanted:?} missing:\n{}",
                page.body
            );
        }
        Ok(())
    }

    /// The rest of org's structure, which the wiki shows exactly as often as
    /// the corpus uses it: example blocks, rules, strikethrough, a forced line
    /// break, a quote with a paragraph break in it, and a list item carrying
    /// more than one line.
    #[test]
    fn the_remaining_org_structure_renders() -> TestResult {
        let org = "\
#+begin_example
literal
#+end_example

-----

+struck+ text with a
continued line.

#+begin_quote
first

second
#+end_quote

- an item
  whose text continues
";
        let stream = crate::record::normalize(ArtifactKind::Note, org.as_bytes())?;
        let page = render(
            &BlobHash::new("c0ffee")?,
            &header(ArtifactKind::Note)?,
            &stream,
            &Directory::default(),
        );
        for wanted in [
            "```\nliteral\n```",
            "---\n",
            "~~struck~~",
            "> first",
            ">\n",
            "> second",
            "- an item",
        ] {
            assert!(
                page.body.contains(wanted),
                "{wanted:?} missing:\n{}",
                page.body
            );
        }
        Ok(())
    }

    /// A `#custom-id` reference points inside the document that wrote it, not
    /// at another record, and is 172 of the real corpus's 2,921 name links.
    /// Reporting those as unresolvable would put a quarter of the projection's
    /// link report on targets that were never corpus references at all.
    #[test]
    fn an_in_page_anchor_stays_an_anchor_and_is_not_reported() -> TestResult {
        let stream = crate::record::normalize(
            ArtifactKind::Note,
            b"* One\n\nsee [[#glossary][the glossary]].\n",
        )?;
        let page = render(
            &BlobHash::new("c0ffee")?,
            &header(ArtifactKind::Note)?,
            &stream,
            &Directory::default(),
        );
        assert!(
            page.body.contains("[the glossary](#glossary)"),
            "{}",
            page.body
        );
        assert!(page.unresolved.is_empty(), "{:?}", page.unresolved);
        Ok(())
    }

    /// A stream whose payload does not end in a newline still produces a page
    /// that does, because a file without a trailing newline is a diff every
    /// tool reports forever.
    #[test]
    fn a_page_always_ends_with_a_newline() -> TestResult {
        let stream = NormalizedStream::from_bytes(b"kb-stream/3\nno trailing newline".to_vec());
        let page = render(
            &BlobHash::new("c0ffee")?,
            &header(ArtifactKind::MailMessage)?,
            &stream,
            &Directory::default(),
        );
        assert!(
            page.body.ends_with("no trailing newline\n"),
            "{}",
            page.body
        );
        Ok(())
    }

    /// A space that cannot be created is reported with the path in it, not
    /// swallowed into a projection that claims to have written nothing.
    #[test]
    fn a_space_that_cannot_be_written_is_reported() -> TestResult {
        let dir = tempfile::tempdir()?;
        let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
        store_alpha(&store)?;
        let space = dir.path().join("space");
        std::fs::create_dir_all(&space)?;
        // `kb/` is where the corpus's pages go; a file of that name makes
        // creating the directory impossible.
        std::fs::write(space.join("kb"), "in the way\n")?;

        let error = project(&store, &space, false).unwrap_err();
        assert!(
            matches!(&error, ProjectError::Space { path, .. } if path.ends_with("kb")),
            "{error:?}"
        );
        Ok(())
    }

    /// A directory sitting where a page belongs is an error rather than a
    /// silent skip: something else owns that name and kb cannot project there.
    #[test]
    fn a_directory_where_a_page_belongs_is_reported() -> TestResult {
        let dir = tempfile::tempdir()?;
        let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
        store_alpha(&store)?;
        let space = dir.path().join("space");
        std::fs::create_dir_all(space.join("kb/alpha.md"))?;

        let error = project(&store, &space, false).unwrap_err();
        assert!(matches!(&error, ProjectError::Space { .. }), "{error:?}");
        Ok(())
    }

    /// The orphan scan walks subdirectories and ignores everything that is not
    /// a marked markdown page, because the space is the operator's directory
    /// and may hold their own files.
    #[test]
    fn the_orphan_scan_reads_subdirectories_and_skips_foreign_files() -> TestResult {
        let dir = tempfile::tempdir()?;
        let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
        store_alpha(&store)?;
        let space = dir.path().join("space");
        std::fs::create_dir_all(space.join("notes/deeper"))?;
        std::fs::write(space.join("notes/deeper/theirs.md"), "# mine\n")?;
        std::fs::write(space.join("notes/data.json"), "{}\n")?;

        let done = project(&store, &space, true)?;
        assert!(done.orphaned.is_empty(), "{:?}", done.orphaned);
        assert_eq!(done.pruned, 0);
        assert!(space.join("notes/deeper/theirs.md").exists());
        Ok(())
    }

    /// One note in the store, for the tests that care about the filesystem
    /// rather than about what a record renders to.
    fn store_alpha(store: &GitBlobStore) -> TestResult {
        let org = "#+title: Alpha\n\nprose.\n";
        let normalized = crate::record::normalize(ArtifactKind::Note, org.as_bytes())?;
        let head = RecordHeader {
            raw: ContentHash::new(store.put(org.as_bytes())?.as_str())?,
            stream: ContentHash::new(store.put(normalized.as_bytes())?.as_str())?,
            ..header(ArtifactKind::Note)?
        };
        let hash = store.put(&head.serialize())?;
        store.set_ref(&RefName::new("kb/alpha")?, &hash)?;
        Ok(())
    }

    /// A code block containing a fence needs a longer fence, or the page ends
    /// mid-block and everything after it renders as prose.
    #[test]
    fn a_fence_is_longer_than_the_content_it_holds() {
        assert_eq!(longest_backtick_run("a ``` b"), 3);
        assert_eq!(longest_backtick_run("none"), 0);
    }

    /// The marker is read out of frontmatter and nowhere else, so a page whose
    /// body merely mentions the key is not mistaken for one kb wrote.
    #[test]
    fn the_marker_is_read_from_the_frontmatter() {
        let page = format!("---\ntitle: \"x\"\n{MARKER_HASH}abc123\n---\n\nbody\n");
        assert_eq!(marker_hash(&page), Some("abc123"));
        assert_eq!(marker_hash("# just a page\n"), None);
    }

    #[test]
    fn a_yaml_scalar_escapes_its_delimiters() {
        assert_eq!(
            yaml_escape(r#"a "quoted" \ thing"#),
            r#"a \"quoted\" \\ thing"#
        );
    }

    /// An alias that would break the wikilink is dropped rather than escaped.
    #[test]
    fn a_wikilink_drops_an_unusable_alias() {
        assert_eq!(wikilink("kb/beta", "[]|"), "[[kb/beta]]");
        assert_eq!(wikilink("kb/beta", "kb/beta"), "[[kb/beta]]");
        assert_eq!(wikilink("kb/beta", "other"), "[[kb/beta|other]]");
    }

    #[test]
    fn a_page_outside_the_space_has_no_name() {
        assert_eq!(
            page_name_of(
                std::path::Path::new("/space"),
                std::path::Path::new("/elsewhere/a.md")
            ),
            None
        );
    }

    /// A store whose record blob is not a record is reported by name rather
    /// than by a bare parse failure (ENG-004).
    #[test]
    fn a_blob_that_is_not_a_record_is_reported_with_its_ref() -> TestResult {
        let dir = tempfile::tempdir()?;
        let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
        let hash = store.put(b"not a record at all\n")?;
        store.set_ref(&RefName::new("kb/broken")?, &hash)?;
        let error = project(&store, &dir.path().join("space"), false).unwrap_err();
        assert!(
            matches!(&error, ProjectError::Record { record, .. } if record == "kb/broken"),
            "{error:?}"
        );
        assert!(error.to_string().contains("kb/broken"), "{error}");
        Ok(())
    }

    /// A record whose stream blob is missing names the record, not the hash.
    #[test]
    fn a_missing_stream_is_reported_against_its_record() -> TestResult {
        let dir = tempfile::tempdir()?;
        let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
        let hash = store.put(&header(ArtifactKind::Note)?.serialize())?;
        store.set_ref(&RefName::new("kb/alpha")?, &hash)?;
        let error = project(&store, &dir.path().join("space"), false).unwrap_err();
        assert!(
            matches!(&error, ProjectError::Store { record, .. } if record == "alpha"),
            "{error:?}"
        );
        Ok(())
    }
}
