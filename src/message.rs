//! The readable text of one mail message.
//!
//! Delivered bytes are not a message's text. In a sample of non-bulk
//! correspondence, only about one message in forty was plain `text/plain`;
//! the rest were multipart or HTML, and many were quoted-printable. Indexing
//! the bytes as they arrived would put MIME boundaries, markup and base64
//! attachments into the embedded passage for nearly all of the corpus.
//!
//! MIME is wrapped rather than threaded through the codebase, per ENG-010:
//! this module is the only place that knows `mail_parser` exists, and its
//! surface is one function from bytes to text. That keeps the choice of
//! parser reversible and keeps the rest of the mail path working on strings.
//!
//! The bias matches the discriminant's. Anything unreadable is kept rather
//! than dropped — a message the parser cannot make sense of is still
//! correspondence, and an empty passage would remove it from the index
//! without saying so.

use mail_parser::MessageParser;

/// Headers the normalized stream emits, in order.
///
/// Who wrote, to whom, when and about what are the things a question names.
/// Everything else in a header block is transport — `Received` chains, DKIM
/// signatures, mailer fingerprints — and would swamp a short message's own
/// words in the embedded text.
///
/// `In-Reply-To` and `References` are read from every message and emitted by
/// none. They are thread-assembly inputs, consumed before normalization, and
/// measured over the first increment they carry no retrievable content while
/// costing real space where it is scarcest: 45% of message passages had a
/// `References` line averaging some 340 characters, sitting ahead of the body
/// — inside the reranker's 2,000-character window and at the front of what
/// the embedder reads. `Message-ID` stays, because it is one line and it is
/// what makes the stream say which message it is.
///
/// The order is the normalizer's, not the sender's: the same message
/// delivered with its fields rearranged is the same record.
const STREAM_HEADERS: [&str; 6] = ["from", "to", "cc", "date", "subject", "message-id"];

/// One message, read.
///
/// The typed form exists so callers take fields rather than re-parsing text:
/// thread assembly wants the identifier headers, the index wants the text,
/// and neither should be grepping a rendered string for them (ENG-006).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Readable {
    /// `From`, display name and address together.
    pub from: String,
    /// `To`, comma-separated.
    pub to: String,
    /// `Cc`, comma-separated.
    pub cc: String,
    /// `Date`, canonicalized.
    pub date: String,
    /// `Date` as seconds since the epoch, for ordering.
    ///
    /// Kept beside the rendered form because a thread has to be assembled in
    /// the order it happened, and the rendered date carries an offset that
    /// does not sort. Zero when the message carries no readable date, which
    /// puts undated mail at the start of its thread rather than dropping it.
    pub epoch: i64,
    /// `Subject`, with encoded words decoded.
    pub subject: String,
    /// `Message-ID`: the identity.
    pub message_id: String,
    /// `In-Reply-To`, for thread assembly.
    pub in_reply_to: String,
    /// `References`, for thread assembly.
    pub references: Vec<String>,
    /// The readable body: a plain part in preference to an HTML one, decoded
    /// from whatever transfer encoding and character set it arrived in, with
    /// attachments excluded.
    pub body: String,
}

/// Read one message into its typed form.
///
/// Total rather than fallible, because there is no such thing as mail too
/// malformed to archive. A message the parser cannot make sense of yields its
/// raw text as the body, which is a poor record but an honest one; refusing it
/// would lose correspondence that exists.
#[must_use]
pub fn parse(raw: &[u8]) -> Readable {
    let Some(parsed) = MessageParser::default().parse(raw) else {
        return Readable {
            body: String::from_utf8_lossy(raw).into_owned(),
            ..Readable::default()
        };
    };
    // `body_text` already renders an HTML-only message to text; 788 of the
    // increment's 6,305 messages arrive that way and none of them needs a
    // second markup stripper. A hand-rolled one was written here first and
    // measured as never executing.
    let body = parsed
        .body_text(0)
        .map(std::borrow::Cow::into_owned)
        .unwrap_or_default();
    let body = if body.trim().is_empty() && parsed.subject().is_none() {
        String::from_utf8_lossy(raw).into_owned()
    } else {
        body.trim().to_owned()
    };
    // Line endings are transport, not content: the same message delivered
    // with CRLF and with LF is one record, and leaving the difference in the
    // body would give it two stream hashes and two sets of spans.
    let body = body.replace("\r\n", "\n").replace('\r', "\n");
    Readable {
        from: addresses(parsed.from()),
        to: addresses(parsed.to()),
        cc: addresses(parsed.cc()),
        date: parsed.date().map(ToString::to_string).unwrap_or_default(),
        epoch: parsed.date().map_or(0, mail_parser::DateTime::to_timestamp),
        subject: parsed.subject().unwrap_or_default().to_owned(),
        // Angle brackets are part of how a message identifier is written
        // everywhere else that matters — the ground-truth question set, the
        // catalogue, mu's `msgid:` queries — and a parser that strips them
        // would leave the corpus with two spellings of one identity.
        message_id: bracketed(parsed.message_id().unwrap_or_default()),
        in_reply_to: bracketed(parsed.in_reply_to().as_text().unwrap_or_default()),
        references: parsed
            .references()
            .as_text_list()
            .map(|list| list.iter().map(|r| bracketed(r)).collect())
            .unwrap_or_default(),
        body,
    }
}

impl Readable {
    /// The value of one indexed header, by the name the stream emits.
    #[must_use]
    pub fn field(&self, name: &str) -> String {
        match name {
            "from" => self.from.clone(),
            "to" => self.to.clone(),
            "cc" => self.cc.clone(),
            "date" => self.date.clone(),
            "subject" => self.subject.clone(),
            "message-id" => self.message_id.clone(),
            "in-reply-to" => self.in_reply_to.clone(),
            "references" => self.references.join(" "),
            _ => String::new(),
        }
    }

    /// The canonical rendering: kept fields in a fixed order, a blank line,
    /// then the body.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        for name in STREAM_HEADERS {
            let value = self.field(name);
            if value.trim().is_empty() {
                continue;
            }
            out.push_str(name);
            out.push_str(": ");
            out.push_str(value.trim());
            out.push('\n');
        }
        out.push('\n');
        out.push_str(&self.body);
        if !self.body.is_empty() {
            out.push('\n');
        }
        out
    }
}

/// The text of a message as the index should hold it.
#[must_use]
pub fn readable(raw: &[u8]) -> String {
    parse(raw).render()
}

/// Every header the normalized stream emits, in order.
#[must_use]
pub const fn stream_headers() -> [&'static str; 6] {
    STREAM_HEADERS
}

/// Render an address list as the text a question would name it by.
///
/// Both the display name and the address are kept: a question may say "the
/// note from Ada" or name the address, and the passage has to answer either.
fn addresses(header: Option<&mail_parser::Address<'_>>) -> String {
    let Some(header) = header else {
        return String::new();
    };
    let mut parts = Vec::new();
    for address in header.iter() {
        let name = address.name().unwrap_or_default().trim();
        let email = address.address().unwrap_or_default().trim();
        match (name.is_empty(), email.is_empty()) {
            (true, true) => {}
            (true, false) => parts.push(email.to_owned()),
            (false, true) => parts.push(name.to_owned()),
            (false, false) => parts.push(format!("{name} <{email}>")),
        }
    }
    parts.join(", ")
}

/// Write a message identifier the way the rest of the corpus writes it.
fn bracketed(id: &str) -> String {
    let id = id.trim();
    if id.is_empty() || (id.starts_with('<') && id.ends_with('>')) {
        return id.to_owned();
    }
    format!("<{id}>")
}
