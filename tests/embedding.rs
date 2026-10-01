//! The embedding vector codec.
//!
//! What remains of a suite that also covered writing vectors into the
//! superseded database's `embeddings` table through stub clients. That table
//! and its write path went with the rest of the retired backend (T029);
//! vectors live in the derived index now, keyed by the span they were
//! computed from, and `tests/write_embed_cli.rs` covers writing them. The
//! codec is unchanged and is still the thing every stored vector passes
//! through.

use kb::embedding::{decode_embedding, encode_embedding};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn codec_round_trip() -> TestResult {
    let v: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, f32::MIN, f32::MAX];
    let bytes = encode_embedding(&v);
    assert_eq!(bytes.len(), 4 * v.len());
    assert_eq!(decode_embedding(&bytes)?, v);
    Ok(())
}

#[test]
fn codec_encode_of_empty_is_empty() -> TestResult {
    let v: Vec<f32> = vec![];
    assert!(encode_embedding(&v).is_empty());
    assert_eq!(decode_embedding(&encode_embedding(&v))?, v);
    Ok(())
}

#[test]
fn codec_decode_rejects_non_multiple_of_four() -> TestResult {
    let bad: Vec<u8> = vec![0; 7];
    let Err(err) = decode_embedding(&bad) else {
        return Err("decode_embedding should reject a non-multiple-of-4 length".into());
    };
    assert!(err.to_string().contains("multiple of 4"), "got: {err}");
    Ok(())
}
