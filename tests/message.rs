//! What the index should consider the text of a mail message.
//!
//! Only a small minority of real correspondence is plain `text/plain`; most is
//! multipart or HTML. Treating the delivered bytes as the message's text would
//! therefore embed MIME boundaries, markup and base64 attachments for nearly
//! all of the corpus — satisfying the letter of "index the message" while defeating
//! its purpose.

use kb::message::readable;

#[test]
fn a_plain_message_keeps_its_body() {
    let raw = b"From: a@example.invalid\n\
                Subject: a plain one\n\
                \n\
                the body text\n";

    let text = readable(raw);

    assert!(text.contains("the body text"), "got: {text}");
}

/// The headers a question is likely to name — who, to whom, when, about what
/// — belong in the indexed text. Everything else is transport.
#[test]
fn the_retrievable_headers_are_kept_and_the_transport_ones_dropped() {
    let raw = b"From: Ada <ada@example.invalid>\n\
                To: reader@example.invalid\n\
                Subject: quarterly numbers\n\
                Date: Mon, 3 Feb 2025 09:00:00 +0000\n\
                Received: from mx1.example.invalid by mx2.example.invalid\n\
                DKIM-Signature: v=1; a=rsa-sha256; d=example.invalid\n\
                \n\
                body\n";

    let text = readable(raw);

    assert!(text.contains("quarterly numbers"));
    assert!(text.contains("ada@example.invalid"));
    assert!(!text.contains("DKIM"), "transport header leaked: {text}");
    assert!(!text.contains("mx1.example.invalid"), "got: {text}");
}

/// `multipart/alternative` carries the same message twice. The plain part is
/// what a human wrote; the HTML part is a rendering of it.
#[test]
fn the_plain_part_of_an_alternative_is_preferred() {
    let raw = b"From: a@example.invalid\n\
                Subject: two ways\n\
                Content-Type: multipart/alternative; boundary=\"BOUND\"\n\
                \n\
                --BOUND\n\
                Content-Type: text/plain; charset=utf-8\n\
                \n\
                the plain wording\n\
                --BOUND\n\
                Content-Type: text/html; charset=utf-8\n\
                \n\
                <html><body><p>the plain wording</p></body></html>\n\
                --BOUND--\n";

    let text = readable(raw);

    assert!(text.contains("the plain wording"));
    assert!(!text.contains("<p>"), "markup leaked: {text}");
    assert!(!text.contains("BOUND"), "boundary leaked: {text}");
}

/// 788 messages in the increment are HTML with no plain alternative.
/// Discarding them would drop an eighth of the corpus.
#[test]
fn an_html_only_message_is_reduced_to_its_text() {
    let raw = b"From: a@example.invalid\n\
                Subject: html only\n\
                Content-Type: text/html; charset=utf-8\n\
                \n\
                <html><body><p>the readable sentence</p>\
                <style>p { color: red }</style></body></html>\n";

    let text = readable(raw);

    assert!(text.contains("the readable sentence"), "got: {text}");
    assert!(!text.contains("<p>"), "markup leaked: {text}");
    assert!(!text.contains("color: red"), "stylesheet leaked: {text}");
}

/// 768 messages in the increment are quoted-printable. Left encoded, every
/// apostrophe and accented character becomes `=E2=80=99` noise in the
/// embedded text.
#[test]
fn quoted_printable_is_decoded() {
    let raw = b"From: a@example.invalid\n\
                Subject: encoded\n\
                Content-Type: text/plain; charset=utf-8\n\
                Content-Transfer-Encoding: quoted-printable\n\
                \n\
                it=E2=80=99s decoded\n";

    let text = readable(raw);

    assert!(text.contains("it\u{2019}s decoded"), "got: {text}");
    assert!(!text.contains("=E2"), "got: {text}");
}

#[test]
fn base64_is_decoded() {
    let raw = b"From: a@example.invalid\n\
                Subject: encoded\n\
                Content-Type: text/plain; charset=utf-8\n\
                Content-Transfer-Encoding: base64\n\
                \n\
                dGhlIGRlY29kZWQgYm9keQ==\n";

    let text = readable(raw);

    assert!(text.contains("the decoded body"), "got: {text}");
}

/// An attachment is not correspondence. Embedding a PDF's bytes would fill
/// the passage with noise and the vector with nothing.
#[test]
fn an_attachment_is_not_part_of_the_text() {
    let raw = b"From: a@example.invalid\n\
                Subject: with attachment\n\
                Content-Type: multipart/mixed; boundary=\"BOUND\"\n\
                \n\
                --BOUND\n\
                Content-Type: text/plain\n\
                \n\
                see attached\n\
                --BOUND\n\
                Content-Type: application/pdf; name=\"report.pdf\"\n\
                Content-Disposition: attachment; filename=\"report.pdf\"\n\
                Content-Transfer-Encoding: base64\n\
                \n\
                JVBERi0xLjQKJcfsj6IKNSAwIG9iago8PC9MZW5ndGggNiAwIFI+PgpzdHJlYW0K\n\
                --BOUND--\n";

    let text = readable(raw);

    assert!(text.contains("see attached"));
    assert!(!text.contains("JVBERi"), "attachment bytes leaked: {text}");
    assert!(
        !text.contains("PDF-1.4"),
        "decoded attachment leaked: {text}"
    );
}

/// Subjects arrive RFC 2047-encoded often enough that leaving them encoded
/// would put `=?utf-8?B?...?=` into the indexed text of real correspondence.
#[test]
fn an_encoded_word_subject_is_decoded() {
    let raw = b"From: a@example.invalid\n\
                Subject: =?utf-8?B?cXVhcnRlcmx5IHJlc3VsdHM=?=\n\
                \n\
                body\n";

    let text = readable(raw);

    assert!(text.contains("quarterly results"), "got: {text}");
    assert!(!text.contains("=?utf-8?"), "got: {text}");
}

/// The bias is toward inclusion throughout the mail path: a message the
/// parser cannot make sense of is still correspondence, and returning nothing
/// would remove it from the index silently.
#[test]
fn an_unparsable_message_falls_back_to_its_raw_text() {
    let raw = b"this is not a message at all, just a line of text\n";

    let text = readable(raw);

    assert!(
        text.contains("not a message at all"),
        "an unreadable message must not become an empty passage: {text}"
    );
}

/// Mail is not reliably UTF-8 even when a folder's messages happen to be.
#[test]
fn a_latin1_body_is_transcoded_rather_than_refused() {
    let mut raw = b"From: a@example.invalid\n\
                    Subject: accents\n\
                    Content-Type: text/plain; charset=iso-8859-1\n\
                    \n"
    .to_vec();
    // A latin-1 e-acute, written as the byte it arrives as.
    raw.extend_from_slice(&[
        0x63, 0x61, 0x66, 0xe9, b' ', b's', b'o', b'c', b'i', b'e', b't', b'y', b'\n',
    ]);

    let text = readable(&raw);

    assert!(text.contains("café society"), "got: {text}");
}

/// Empty is a legitimate body — a message with only a subject line — and must
/// not be mistaken for a parse failure.
#[test]
fn a_message_with_no_body_still_yields_its_headers() {
    let raw = b"From: a@example.invalid\nSubject: no body here\n\n";

    let text = readable(raw);

    assert!(text.contains("no body here"), "got: {text}");
}

/// Thread assembly reads these two headers, so they have to survive parsing
/// in the bracketed spelling the rest of the corpus uses.
#[test]
fn the_thread_identifiers_are_kept_in_the_corpus_spelling() {
    let raw = b"From: a@example.invalid\n\
                Message-ID: <c@example.invalid>\n\
                In-Reply-To: <b@example.invalid>\n\
                References: <a@example.invalid> <b@example.invalid>\n\
                \n\
                a reply\n";

    let message = kb::message::parse(raw);

    assert_eq!(message.message_id, "<c@example.invalid>");
    assert_eq!(message.in_reply_to, "<b@example.invalid>");
    assert_eq!(
        message.references,
        vec![
            "<a@example.invalid>".to_owned(),
            "<b@example.invalid>".to_owned()
        ]
    );
}

/// The stream's field order is the normalizer's contract: two deliveries of
/// one message must render identically, and a caller reading a field by name
/// must get the same string the stream emitted.
#[test]
fn every_emitted_header_is_reachable_by_name() {
    let raw = b"From: Ada <ada@example.invalid>\n\
                To: Bo <bo@example.invalid>\n\
                Cc: Cy <cy@example.invalid>\n\
                Date: Mon, 3 Feb 2025 09:00:00 +0000\n\
                Subject: all fields\n\
                Message-ID: <c@example.invalid>\n\
                In-Reply-To: <b@example.invalid>\n\
                References: <a@example.invalid>\n\
                \n\
                body\n";
    let message = kb::message::parse(raw);
    let rendered = message.render();

    for name in kb::message::stream_headers() {
        let value = message.field(name);
        assert!(!value.is_empty(), "{name} was empty");
        assert!(
            rendered.contains(&format!("{name}: {value}")),
            "{name} was not emitted as rendered: {rendered}"
        );
    }
    assert_eq!(
        message.field("received"),
        "",
        "a header the stream does not emit has no value to give"
    );
    // Read from every message, emitted by none: thread assembly consumes
    // them, and in the passage they are identifier tokens crowding out the
    // words a question is asked in.
    assert!(!message.in_reply_to.is_empty());
    assert!(!message.references.is_empty());
    assert!(!rendered.contains("in-reply-to:"), "rendered: {rendered}");
    assert!(!rendered.contains("references:"), "rendered: {rendered}");
}

/// A question may name a correspondent either way, so both spellings are kept
/// when both exist and neither is invented when only one does.
#[test]
fn an_address_keeps_whichever_of_name_and_mailbox_it_has() {
    let named = kb::message::parse(b"From: Ada <ada@example.invalid>\n\nbody\n");
    assert_eq!(named.from, "Ada <ada@example.invalid>");

    let bare = kb::message::parse(b"From: ada@example.invalid\n\nbody\n");
    assert_eq!(bare.from, "ada@example.invalid");

    let absent = kb::message::parse(b"Subject: no sender\n\nbody\n");
    assert_eq!(absent.from, "");
}

/// Several correspondents are one field, and the rendering has to keep them
/// separable.
#[test]
fn multiple_recipients_are_rendered_as_a_list() {
    let message = kb::message::parse(
        b"From: a@example.invalid\n\
          To: Bo <bo@example.invalid>, cy@example.invalid\n\
          \n\
          body\n",
    );

    assert_eq!(message.to, "Bo <bo@example.invalid>, cy@example.invalid");
}

/// Line endings are transport. The same message delivered with CRLF and with
/// LF is one record, and a difference here would give it two stream hashes.
#[test]
fn line_endings_do_not_change_the_rendering() {
    let crlf = kb::message::readable(
        b"From: a@example.invalid\r\nSubject: s\r\n\r\nline one\r\nline two\r\n",
    );
    let lf = kb::message::readable(b"From: a@example.invalid\nSubject: s\n\nline one\nline two\n");

    assert_eq!(crlf, lf);
    assert!(!crlf.contains('\r'), "carriage return survived: {crlf:?}");
}

/// An empty body is a legitimate message, and its rendering must not grow a
/// trailing blank line that a body-bearing message would not have.
#[test]
fn a_bodyless_message_renders_without_a_dangling_line() {
    let message = kb::message::parse(b"From: a@example.invalid\nSubject: ping\n\n");

    let rendered = message.render();

    assert!(rendered.ends_with("\n\n"), "rendered: {rendered:?}");
    assert!(rendered.contains("subject: ping"));
}
