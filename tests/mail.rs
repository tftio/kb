//! Behavior of the bulk/non-bulk discriminant.
//!
//! The rule decides what enters the corpus, so its errors are not symmetric.
//! Retaining a newsletter costs a little noise. Dropping real correspondence
//! removes ground truth silently — the exact failure this plan exists to
//! prevent — so the rule is biased toward inclusion and every test here is
//! written from that asymmetry.

use kb::mail::{BulkRule, Classification};

fn rule() -> BulkRule {
    BulkRule::default()
}

/// The four header families that mark automated mail: List-Id,
/// List-Unsubscribe, Precedence, and Auto-Submitted.
#[test]
fn a_message_carrying_a_bulk_header_is_classified_bulk() {
    for header in [
        "List-Unsubscribe: <mailto:x@y>",
        "List-Id: <announce.example.com>",
        "Precedence: bulk",
        "Auto-Submitted: auto-generated",
    ] {
        let message = format!("From: a@b\nTo: reader@example.invalid\n{header}\n\nbody\n");
        assert!(
            matches!(rule().classify(&message), Classification::Bulk { .. }),
            "not classified bulk: {header}"
        );
    }
}

/// Ordinary correspondence carries none of them.
#[test]
fn a_message_from_a_person_is_classified_non_bulk() {
    let message = "From: Ada <ada@example.com>\nTo: reader@example.invalid\n\
                   Subject: the schema change\n\nShall we cut it?\n";

    assert_eq!(rule().classify(message), Classification::NonBulk);
}

/// Header names are case-insensitive per RFC 5322, and real mail is
/// inconsistent about it.
#[test]
fn header_matching_ignores_case() {
    let message = "From: a@b\nLIST-ID: <x.example.com>\n\nbody\n";

    assert!(matches!(
        rule().classify(message),
        Classification::Bulk { .. }
    ));
}

/// A bulk-looking string in the body is not a header. Without this the rule
/// would drop any correspondence that quotes a newsletter.
#[test]
fn a_bulk_signal_in_the_body_is_not_a_header() {
    let message = "From: Ada <ada@example.com>\nTo: reader@example.invalid\n\n\
                   I got this: List-Unsubscribe: <mailto:x@y>\nWhat is it?\n";

    assert_eq!(rule().classify(message), Classification::NonBulk);
}

/// The bias, stated as a test: anything the rule cannot read is kept. A
/// message with no header block at all is malformed, not automated.
#[test]
fn an_unreadable_message_is_kept() {
    assert_eq!(rule().classify(""), Classification::NonBulk);
    assert_eq!(
        rule().classify("not a message at all"),
        Classification::NonBulk
    );
}

/// The classification names which signal fired, so a false negative can be
/// traced to the rule that caused it rather than guessed at.
#[test]
fn a_bulk_classification_names_the_signal_that_fired() {
    let message = "From: a@b\nPrecedence: bulk\n\nbody\n";

    let outcome = rule().classify(message);

    assert!(
        matches!(&outcome, Classification::Bulk { signal } if signal == "precedence"),
        "expected the precedence signal, got {outcome:?}"
    );
}

/// The rule is data, not code: T018's invariant requires it to be re-tunable
/// and re-measurable without a code change, because its error rate is the
/// deliverable and tuning is how that rate moves.
#[test]
fn the_signal_set_is_configuration() {
    let narrow = BulkRule::from_signals(["list-id"]);
    let message = "From: a@b\nPrecedence: bulk\n\nbody\n";

    assert_eq!(
        narrow.classify(message),
        Classification::NonBulk,
        "a signal outside the configured set must not classify"
    );
    assert!(matches!(
        BulkRule::from_signals(["precedence"]).classify(message),
        Classification::Bulk { .. }
    ));
}

/// Folded headers continue onto following lines. A continuation is part of
/// the value above it, never a new header, so a folded value that happens to
/// begin with a signal name must not fire.
#[test]
fn a_folded_continuation_is_not_a_new_header() {
    let message = "From: a@b\nSubject: a long subject\n List-Id: not a header\n\nbody\n";

    assert_eq!(rule().classify(message), Classification::NonBulk);
}
