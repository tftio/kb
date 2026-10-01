//! Behavior of the stored record: its normalized stream, its self-describing
//! header, and the index row both must be reconstructible from.

use kb::record::{ArtifactKind, PassageLevel, normalize, passages};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const NOTE: &[u8] = b"* Against Solipsism\n\nThe argument turns on   whitespace.\n";

/// Spans address the normalized stream by byte offset, so the same input
/// producing different bytes on a second run would silently repoint every
/// span derived from the first.
#[test]
fn normalizing_twice_yields_identical_bytes() -> TestResult {
    let once = normalize(ArtifactKind::Note, NOTE)?;
    let twice = normalize(ArtifactKind::Note, NOTE)?;

    assert_eq!(once.as_bytes(), twice.as_bytes());
    Ok(())
}

/// Mail arrives with fields in arbitrary order, values folded across lines,
/// CRLF endings, and a long tail of headers added by every relay it passed
/// through. Normalizing collapses all of that, so two deliveries of the same
/// message hash the same and a span means the same thing in both.
#[test]
fn a_mail_message_normalizes_to_canonical_headers_and_body() -> TestResult {
    let raw = b"X-Spam-Score: 3.1\r\n\
                Subject: Re: the\r\n schema change\r\n\
                From: Ada <ada@example.com>\r\n\
                Date: Thu, 14 Aug 2026 09:00:00 +0000\r\n\
                Message-ID: <m1@example.com>\r\n\
                To: Bo <bo@example.com>\r\n\
                \r\n\
                Body first line.\r\n\
                Body second line.\r\n";

    let stream = normalize(ArtifactKind::MailMessage, raw)?;
    let text = String::from_utf8(stream.as_bytes().to_vec())?;

    assert_eq!(
        text,
        "kb-stream/3\n\
         from: Ada <ada@example.com>\n\
         to: Bo <bo@example.com>\n\
         date: 2026-08-14T09:00:00Z\n\
         subject: Re: the schema change\n\
         message-id: <m1@example.com>\n\
         \n\
         Body first line.\n\
         Body second line.\n",
        "normalized mail was:\n{text}"
    );
    Ok(())
}

/// The header order is the normalizer's, not the sender's: the same message
/// delivered with its fields in a different order is the same record.
#[test]
fn mail_header_order_is_the_normalizers_not_the_senders() -> TestResult {
    let one = b"From: a@x\r\nTo: b@x\r\nSubject: s\r\n\r\nbody\r\n";
    let other = b"Subject: s\r\nTo: b@x\r\nFrom: a@x\r\n\r\nbody\r\n";

    assert_eq!(
        normalize(ArtifactKind::MailMessage, one)?.as_bytes(),
        normalize(ArtifactKind::MailMessage, other)?.as_bytes()
    );
    Ok(())
}

/// The stream carries the version of the normalizer that produced it, so a
/// parser fix invalidates the spans computed under the old interpretation
/// instead of leaving them pointing at differently-read content.
#[test]
fn the_stream_names_the_normalizer_that_produced_it() -> TestResult {
    let stream = normalize(ArtifactKind::Note, NOTE)?;

    assert!(
        stream.as_bytes().starts_with(b"kb-stream/"),
        "stream did not open with its version: {:?}",
        String::from_utf8_lossy(stream.as_bytes().get(..20).unwrap_or(stream.as_bytes()))
    );
    Ok(())
}

/// A transcript's passages are its turns. Splitting a conversation on a
/// character budget cuts mid-answer and embeds half an argument; the turn is
/// the unit a question is actually answered in, and the org headings the
/// importer writes already mark it.
#[test]
fn a_transcript_breaks_on_turn_boundaries() -> TestResult {
    let raw = b"* Human [2026-08-14 09:00]\n\nHow did we set up CleanShotX?\n\n\
                * Assistant [2026-08-14 09:01]\n\nThrough the Nix module.\n";
    let stream = normalize(ArtifactKind::SessionTranscript, raw)?;

    let found = passages(ArtifactKind::SessionTranscript, &stream);

    assert_eq!(found.len(), 2, "one passage per turn, got: {found:?}");
    assert!(found.iter().all(|p| p.level == PassageLevel::Turn));
    let first = nth(&stream, &found, 0);
    let second = nth(&stream, &found, 1);
    assert!(first.starts_with("* Human"), "first turn was: {first}");
    assert!(first.contains("CleanShotX"), "first turn was: {first}");
    assert!(
        second.starts_with("* Assistant"),
        "second turn was: {second}"
    );
    assert!(
        !first.contains("Nix module"),
        "the assistant's turn leaked into the human's: {first}"
    );
    Ok(())
}

/// An authored note breaks at its sections, for the same reason and by the
/// same mechanism, but the unit means something different and is labelled so.
#[test]
fn a_note_breaks_on_section_boundaries() -> TestResult {
    let raw = b"* Background\n\nWhat came before.\n\n* Decision\n\nWhat was chosen.\n";
    let stream = normalize(ArtifactKind::Note, raw)?;

    let found = passages(ArtifactKind::Note, &stream);

    assert_eq!(found.len(), 2, "one passage per section, got: {found:?}");
    assert!(found.iter().all(|p| p.level == PassageLevel::Section));
    assert!(nth(&stream, &found, 1).starts_with("* Decision"));
    Ok(())
}

/// A single message is one passage: it is already the unit correspondence
/// arrives in.
#[test]
fn a_mail_message_is_one_passage() -> TestResult {
    let raw = b"From: a@x\r\nSubject: s\r\n\r\nthe body.\r\n";
    let stream = normalize(ArtifactKind::MailMessage, raw)?;

    let found = passages(ArtifactKind::MailMessage, &stream);

    assert_eq!(found.len(), 1);
    assert!(
        found
            .first()
            .is_some_and(|p| p.level == PassageLevel::Message),
        "levels were: {found:?}"
    );
    assert!(nth(&stream, &found, 0).contains("the body."));
    Ok(())
}

/// A thread is indexed at both levels: the whole exchange, because a decision
/// is often only visible across it, and each message, because that is what a
/// citation has to point at.
#[test]
fn a_thread_is_indexed_at_thread_and_message_level() -> TestResult {
    let raw = b"From alice@example.com Thu Aug 14 09:00:00 2026\r\n\
                From: alice@example.com\r\nSubject: schema\r\n\r\nShall we cut it?\r\n\
                From bo@example.com Thu Aug 14 09:30:00 2026\r\n\
                From: bo@example.com\r\nSubject: Re: schema\r\n\r\nYes, cut it.\r\n";
    let stream = normalize(ArtifactKind::MailThread, raw)?;

    let found = passages(ArtifactKind::MailThread, &stream);

    let threads = found
        .iter()
        .filter(|p| p.level == PassageLevel::Thread)
        .count();
    let messages: Vec<kb::record::Passage> = found
        .iter()
        .filter(|p| p.level == PassageLevel::Message)
        .cloned()
        .collect();
    assert_eq!(threads, 1, "one thread-level passage, got: {found:?}");
    assert_eq!(messages.len(), 2, "one passage per message, got: {found:?}");
    assert!(nth(&stream, &messages, 0).contains("Shall we cut it?"));
    assert!(nth(&stream, &messages, 1).contains("Yes, cut it."));
    assert!(
        !nth(&stream, &messages, 0).contains("Yes, cut it."),
        "the reply leaked into the first message"
    );
    Ok(())
}

/// Every span must address bytes that are actually there. A span that runs
/// past the end of the stream is a citation to nothing, which is the failure
/// mode the whole provenance argument exists to prevent.
#[test]
fn every_passage_addresses_bytes_within_its_stream() -> TestResult {
    let cases: [(ArtifactKind, &[u8]); 4] = [
        (ArtifactKind::Note, b"* A\n\nbody.\n"),
        (
            ArtifactKind::SessionTranscript,
            b"* Human [t]\n\nq\n\n* Assistant [t]\n\na\n",
        ),
        (ArtifactKind::MailMessage, b"From: a@x\r\n\r\nbody\r\n"),
        (
            ArtifactKind::MailThread,
            b"From a@x Thu Aug 14 09:00:00 2026\r\nFrom: a@x\r\n\r\nbody\r\n",
        ),
    ];
    for (kind, raw) in cases {
        let stream = normalize(kind, raw)?;
        for passage in passages(kind, &stream) {
            assert!(
                passage.span.end() <= stream.as_bytes().len(),
                "{kind} passage {passage:?} runs past a {}-byte stream",
                stream.as_bytes().len()
            );
        }
    }
    Ok(())
}

/// The text of the `index`th passage, or the empty string when there is no
/// such passage — which fails the caller's assertion with its own message
/// rather than panicking here.
fn nth(
    stream: &kb::record::NormalizedStream,
    passages: &[kb::record::Passage],
    index: usize,
) -> String {
    passages.get(index).map_or_else(String::new, |passage| {
        String::from_utf8_lossy(
            stream
                .as_bytes()
                .get(passage.span.start()..passage.span.end())
                .unwrap_or_default(),
        )
        .into_owned()
    })
}

/// The whole point of a self-describing record: hand the index nothing but
/// the stored bytes and it can rebuild its row. If any field were reachable
/// only from the index, the index would be authoritative for it and ST-002
/// would be false.
#[test]
fn an_index_row_is_rebuilt_from_the_record_blob_alone() -> TestResult {
    let header = sample_header()?;
    let at_write_time = header.index_row();

    let blob = header.serialize();
    let rebuilt = kb::record::index_row_from_blob(&blob)?;

    assert_eq!(rebuilt, at_write_time);
    Ok(())
}

/// Tag order, timestamps and hashes all survive the round trip; a field
/// quietly dropped in serialization would show up here rather than as a
/// mystery in the index months later.
#[test]
fn a_header_round_trips_through_its_serialized_form() -> TestResult {
    let header = sample_header()?;

    let parsed = kb::record::RecordHeader::parse(&header.serialize())?;

    assert_eq!(parsed, header);
    Ok(())
}

/// A record that names no tags is ordinary, and must not be confused with one
/// whose tags failed to parse.
#[test]
fn a_record_with_no_tags_round_trips_as_untagged() -> TestResult {
    let mut header = sample_header()?;
    header.tags.clear();

    let parsed = kb::record::RecordHeader::parse(&header.serialize())?;

    assert!(parsed.tags.is_empty(), "tags were: {:?}", parsed.tags);
    Ok(())
}

/// Bytes that are not a record are refused with a reason, rather than parsed
/// into a plausible-looking row.
#[test]
fn bytes_that_are_not_a_record_are_refused() {
    let outcome = kb::record::RecordHeader::parse(b"* Just an org note\n\nnot a record.\n");

    assert!(
        outcome.is_err(),
        "an arbitrary org document must not parse as a record header"
    );
}

/// Links the source itself wrote are facts about the corpus, and each cites
/// the span it was found in so the claim is checkable.
#[test]
fn links_the_author_wrote_are_extracted_with_their_spans() -> TestResult {
    let raw =
        b"* Decision\n\nSee [[storage-plan]] and [[id:b70049ea-001f-49fd-ba5e-4344fbde9d92]].\n";
    let stream = normalize(ArtifactKind::Note, raw)?;

    let links = kb::record::authored_links(&stream);

    assert_eq!(links.len(), 2, "links were: {links:?}");
    let cited: Vec<String> = links
        .iter()
        .map(|l| {
            String::from_utf8_lossy(
                stream
                    .as_bytes()
                    .get(l.span.start()..l.span.end())
                    .unwrap_or_default(),
            )
            .into_owned()
        })
        .collect();
    assert_eq!(
        cited,
        vec![
            "[[storage-plan]]".to_owned(),
            "[[id:b70049ea-001f-49fd-ba5e-4344fbde9d92]]".to_owned()
        ]
    );
    Ok(())
}

fn sample_header() -> Result<kb::record::RecordHeader, Box<dyn std::error::Error>> {
    Ok(kb::record::RecordHeader {
        id: kb::record::RecordId::new("b70049ea-001f-49fd-ba5e-4344fbde9d92")?,
        corpus: kb::record::CorpusId::new("kb")?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: kb::record::SourceRef::new("node-id", "b70049ea-001f-49fd-ba5e-4344fbde9d92")?,
        tags: vec![
            kb::record::Tag::new("storage")?,
            kb::record::Tag::new("retrieval")?,
        ],
        raw: kb::record::ContentHash::new("2aae6c35c94fcfb415dbe95f408b9ce91ee846ed")?,
        stream: kb::record::ContentHash::new("8251c7fb190011b09c1b69a7d74a24571a7d2f10")?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    })
}

/// The spool T022 delivers is not a special case: a hook writes one
/// self-describing file per session and the server ingests it by the same
/// path as anything else. If a spooled session needed its own record shape,
/// capture would be a second ingest path to keep in step with this one.
#[test]
fn a_spooled_session_record_is_an_ordinary_record() -> TestResult {
    let transcript = b"* Human [2026-08-14 09:00]\n\nWhat did we decide?\n\n\
                       * Assistant [2026-08-14 09:01]\n\nTo pack but never prune.\n";
    let stream = normalize(ArtifactKind::SessionTranscript, transcript)?;
    let header = kb::record::RecordHeader {
        kind: ArtifactKind::SessionTranscript,
        source: kb::record::SourceRef::new("session-id", "81fe1574-14de-451e-8e5a-428357d695f6")?,
        ..sample_header()?
    };

    let rebuilt = kb::record::index_row_from_blob(&header.serialize())?;

    assert_eq!(rebuilt, header.index_row());
    assert_eq!(rebuilt.source.scheme(), "session-id");
    assert_eq!(
        passages(ArtifactKind::SessionTranscript, &stream).len(),
        2,
        "a spooled session still breaks on its turns"
    );
    Ok(())
}

proptest::proptest! {
    /// Determinism has to hold for whatever actually arrives, not only for
    /// the inputs a test author thought of: a stream that differs between two
    /// runs would repoint every span derived from the first.
    #[test]
    fn normalizing_any_text_twice_yields_identical_bytes(text in ".{0,400}") {
        for kind in [
            ArtifactKind::Note,
            ArtifactKind::SessionTranscript,
            ArtifactKind::MailMessage,
            ArtifactKind::MailThread,
        ] {
            let once = normalize(kind, text.as_bytes());
            let twice = normalize(kind, text.as_bytes());
            match (once, twice) {
                (Ok(a), Ok(b)) => proptest::prop_assert_eq!(a.as_bytes(), b.as_bytes()),
                (Err(_), Err(_)) => {}
                (a, b) => proptest::prop_assert!(
                    false,
                    "one run succeeded and the other did not: {:?} vs {:?}",
                    a.is_ok(),
                    b.is_ok()
                ),
            }
        }
    }
}

/// A document with no headings still has content, and content that cannot be
/// retrieved may as well not be stored. 14 of the live corpus's 1832 notes
/// are keyword-only documents of exactly this shape -- a `#+title:` and a
/// `#+filetags:` and no `*` heading anywhere -- and they produced no passages
/// at all until this was fixed.
#[test]
fn a_document_with_no_headings_is_one_passage() -> TestResult {
    let raw = b"#+title: Lobsters comments 2025\n#+name: lobsters-index-2025\n\nA body with no heading.\n";
    let stream = normalize(ArtifactKind::Note, raw)?;

    let found = passages(ArtifactKind::Note, &stream);

    assert_eq!(found.len(), 1, "passages were: {found:?}");
    assert!(
        nth(&stream, &found, 0).contains("A body with no heading."),
        "the passage does not cover the body: {}",
        nth(&stream, &found, 0)
    );
    Ok(())
}

/// An empty document is the one case that legitimately has nothing to
/// retrieve, and must not be papered over with an empty passage.
#[test]
fn an_empty_document_has_no_passages() -> TestResult {
    let stream = normalize(ArtifactKind::Note, b"")?;

    assert!(passages(ArtifactKind::Note, &stream).is_empty());
    Ok(())
}
