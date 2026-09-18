//! Exact scanner for the pinned ordered Split regex, followed by `ByteLevel`.

use unicode_general_category::{GeneralCategory, get_general_category};

use crate::{TokenizerError, byte_level};

/// A half-open character-offset span in the caller's UTF-8 text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextSpan {
    pub start: usize,
    pub end: usize,
}

/// One isolated Split piece after the no-regex `ByteLevel` transform.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PretokenizedPiece {
    /// Exact source text before byte mapping.
    pub source: String,
    /// Reversible `ByteLevel` representation passed to BPE merging.
    pub byte_mapped: String,
    /// Character offsets, matching the reference tokenizer API.
    pub span: TextSpan,
}

#[derive(Clone, Copy)]
struct Character {
    value: char,
    byte_start: usize,
    byte_end: usize,
}

/// Applies the packet's ordered Split expression without normalization.
pub(crate) fn pretokenize(input: &str) -> Result<Vec<PretokenizedPiece>, TokenizerError> {
    let characters: Vec<Character> = input
        .char_indices()
        .map(|(byte_start, value)| Character {
            value,
            byte_start,
            byte_end: byte_start + value.len_utf8(),
        })
        .collect();
    let mut output = Vec::new();
    let mut cursor = 0;

    while cursor < characters.len() {
        let end = contraction_end(&characters, cursor)
            .or_else(|| letter_end(&characters, cursor))
            .or_else(|| number_end(&characters, cursor))
            .or_else(|| punctuation_end(&characters, cursor))
            .or_else(|| newline_whitespace_end(&characters, cursor))
            .or_else(|| trailing_whitespace_end(&characters, cursor))
            .or_else(|| whitespace_end(&characters, cursor))
            .ok_or_else(|| TokenizerError::InvalidAsset("Split regex made no progress".into()))?;
        if end <= cursor {
            return Err(TokenizerError::InvalidAsset(
                "Split regex made empty progress".into(),
            ));
        }
        let start_byte = characters[cursor].byte_start;
        let end_byte = characters[end - 1].byte_end;
        let source = input[start_byte..end_byte].to_owned();
        output.push(PretokenizedPiece {
            byte_mapped: byte_level::encode_bytes(source.as_bytes()),
            source,
            span: TextSpan { start: cursor, end },
        });
        cursor = end;
    }

    Ok(output)
}

fn contraction_end(characters: &[Character], start: usize) -> Option<usize> {
    if characters.get(start)?.value != '\'' {
        return None;
    }
    for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
        let suffix: Vec<char> = suffix.chars().collect();
        let end = start + 1 + suffix.len();
        if end <= characters.len()
            && suffix.iter().enumerate().all(|(offset, expected)| {
                contraction_case_matches(characters[start + 1 + offset].value, *expected)
            })
        {
            return Some(end);
        }
    }
    None
}

fn letter_end(characters: &[Character], start: usize) -> Option<usize> {
    let first = characters.get(start)?.value;
    let mut cursor = start;
    if !is_letter(first) {
        if is_carriage_return_or_line_feed(first)
            || is_number(first)
            || start + 1 >= characters.len()
            || !is_letter(characters[start + 1].value)
        {
            return None;
        }
        cursor += 1;
    }
    if !is_letter(characters.get(cursor)?.value) {
        return None;
    }
    while characters
        .get(cursor)
        .is_some_and(|item| is_letter(item.value))
    {
        cursor += 1;
    }
    Some(cursor)
}

fn number_end(characters: &[Character], start: usize) -> Option<usize> {
    if !is_number(characters.get(start)?.value) {
        return None;
    }
    let mut cursor = start;
    while cursor < characters.len() && cursor - start < 3 && is_number(characters[cursor].value) {
        cursor += 1;
    }
    Some(cursor)
}

fn punctuation_end(characters: &[Character], start: usize) -> Option<usize> {
    let mut cursor = start;
    if characters[cursor].value == ' '
        && characters
            .get(cursor + 1)
            .is_some_and(|item| is_punctuation_piece(item.value))
    {
        cursor += 1;
    }
    if !characters
        .get(cursor)
        .is_some_and(|item| is_punctuation_piece(item.value))
    {
        return None;
    }
    while characters
        .get(cursor)
        .is_some_and(|item| is_punctuation_piece(item.value))
    {
        cursor += 1;
    }
    while characters
        .get(cursor)
        .is_some_and(|item| is_carriage_return_or_line_feed(item.value))
    {
        cursor += 1;
    }
    Some(cursor)
}

fn newline_whitespace_end(characters: &[Character], start: usize) -> Option<usize> {
    if !characters
        .get(start)
        .is_some_and(|item| item.value.is_whitespace())
    {
        return None;
    }
    let mut run_end = start;
    let mut last_newline = None;
    while characters
        .get(run_end)
        .is_some_and(|item| item.value.is_whitespace())
    {
        if is_carriage_return_or_line_feed(characters[run_end].value) {
            last_newline = Some(run_end);
        }
        run_end += 1;
    }
    let mut end = last_newline? + 1;
    while characters
        .get(end)
        .is_some_and(|item| is_carriage_return_or_line_feed(item.value))
    {
        end += 1;
    }
    Some(end)
}

fn trailing_whitespace_end(characters: &[Character], start: usize) -> Option<usize> {
    if !characters
        .get(start)
        .is_some_and(|item| item.value.is_whitespace())
    {
        return None;
    }
    let mut end = start;
    while characters
        .get(end)
        .is_some_and(|item| item.value.is_whitespace())
    {
        end += 1;
    }
    if end == characters.len() {
        return Some(end);
    }
    (end > start + 1).then_some(end - 1)
}

fn whitespace_end(characters: &[Character], start: usize) -> Option<usize> {
    if !characters
        .get(start)
        .is_some_and(|item| item.value.is_whitespace())
    {
        return None;
    }
    let mut end = start + 1;
    while characters
        .get(end)
        .is_some_and(|item| item.value.is_whitespace())
    {
        end += 1;
    }
    Some(end)
}

fn contraction_case_matches(actual: char, expected: char) -> bool {
    actual.eq_ignore_ascii_case(&expected) || (expected == 's' && actual == 'ſ')
}
fn is_letter(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

fn is_number(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

fn is_carriage_return_or_line_feed(character: char) -> bool {
    matches!(character, '\r' | '\n')
}

fn is_punctuation_piece(character: char) -> bool {
    !character.is_whitespace() && !is_letter(character) && !is_number(character)
}

#[cfg(test)]
mod tests {
    use super::pretokenize;

    fn source_pieces(input: &str) -> Vec<String> {
        pretokenize(input)
            .unwrap_or_else(|error| panic!("synthetic pretokenization failed: {error}"))
            .into_iter()
            .map(|piece| piece.source)
            .collect()
    }

    #[test]
    fn split_order_preserves_contractions_numbers_and_space_before_words() {
        assert_eq!(
            source_pieces("  I'M 1234!?\r\n"),
            [" ", " I", "'M", " ", "123", "4", "!?\r\n"]
        );
    }

    #[test]
    fn unicode_letters_numbers_marks_and_offsets_are_distinct() {
        let pieces = pretokenize("é e\u{301} Ⅻ")
            .unwrap_or_else(|error| panic!("synthetic pretokenization failed: {error}"));
        assert_eq!(
            pieces
                .iter()
                .map(|piece| piece.source.as_str())
                .collect::<Vec<_>>(),
            ["é", " e", "\u{301}", " ", "Ⅻ"]
        );
        assert_eq!(pieces[2].span.start, 3);
        assert_eq!(pieces[2].span.end, 4);
    }

    #[test]
    fn whitespace_lookahead_leaves_one_space_for_a_following_word() {
        assert_eq!(source_pieces("  Hello  "), [" ", " Hello", "  "]);
    }
}

#[cfg(test)]
mod unicode_profile_oracle {
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    use serde_json::Value;
    use sha2::{Digest, Sha256};

    use super::{Character, contraction_end, is_letter, is_number};

    const MANIFEST_SHA256: &str =
        "5733fc9b82026fd76eb7ac95c50daed5da78adc5c87ceb70a6bf1deacfb7838b";
    const PROFILE_SHA256: &str = "394fae1e7f641ab1e644207482894a0dca218bb59aa7df4294e1bc796124e089";

    #[test]
    #[ignore = "requires hash-verified tokenizer-unicode-profile-001 root"]
    fn every_valid_unicode_scalar_matches_the_pinned_profile() {
        let root = env::var_os("MINIFIELD_TEXT_TOKENIZER_UNICODE_PROFILE_ROOT")
            .map(PathBuf::from)
            .filter(|path| path.is_dir())
            .unwrap_or_else(|| {
                panic!(
                    "MINIFIELD_TEXT_TOKENIZER_UNICODE_PROFILE_ROOT must name the fixture directory"
                )
            });
        let manifest = read_hashed(&root.join("manifest.json"), MANIFEST_SHA256);
        assert!(!manifest.is_empty(), "Unicode fixture manifest is empty");
        let profile: Value =
            serde_json::from_slice(&read_hashed(&root.join("profile.json"), PROFILE_SHA256))
                .unwrap_or_else(|error| panic!("Unicode profile JSON parse failed: {error}"));
        let ranges = profile["inclusive_ranges"]
            .as_object()
            .unwrap_or_else(|| panic!("Unicode profile lacks inclusive_ranges"));
        let letter_ranges = parse_ranges(
            ranges
                .get("letter")
                .unwrap_or_else(|| panic!("Unicode profile lacks letter ranges")),
        );
        let number_ranges = parse_ranges(
            ranges
                .get("number")
                .unwrap_or_else(|| panic!("Unicode profile lacks number ranges")),
        );
        let space_ranges = parse_ranges(
            ranges
                .get("space")
                .unwrap_or_else(|| panic!("Unicode profile lacks space ranges")),
        );
        let mut counts = [0_u64; 3];
        for scalar in 0..=0x10_FFFF {
            let Some(character) = char::from_u32(scalar) else {
                continue;
            };
            let expected_letter = in_ranges(scalar, &letter_ranges);
            let expected_number = in_ranges(scalar, &number_ranges);
            let expected_space = in_ranges(scalar, &space_ranges);
            assert_eq!(
                is_letter(character),
                expected_letter,
                "letter scalar U+{scalar:04X}"
            );
            assert_eq!(
                is_number(character),
                expected_number,
                "number scalar U+{scalar:04X}"
            );
            assert_eq!(
                character.is_whitespace(),
                expected_space,
                "whitespace scalar U+{scalar:04X}"
            );
            counts[0] += u64::from(expected_letter);
            counts[1] += u64::from(expected_number);
            counts[2] += u64::from(expected_space);
        }
        assert_eq!(counts, [141_028, 1_911, 25]);

        let contractions = profile["contraction_fold_cases"]
            .as_array()
            .unwrap_or_else(|| panic!("Unicode profile lacks contraction_fold_cases"));
        for case in contractions {
            let text = case["text"]
                .as_str()
                .unwrap_or_else(|| panic!("contraction case lacks text"));
            let expected_match = case["remainder"]
                .as_array()
                .unwrap_or_else(|| panic!("contraction case lacks remainder"))
                .is_empty();
            let characters: Vec<Character> = text
                .char_indices()
                .map(|(byte_start, value)| Character {
                    value,
                    byte_start,
                    byte_end: byte_start + value.len_utf8(),
                })
                .collect();
            assert_eq!(
                contraction_end(&characters, 0).is_some(),
                expected_match,
                "contraction case {text:?}"
            );
        }
    }

    fn read_hashed(path: &PathBuf, expected: &str) -> Vec<u8> {
        let bytes = fs::read(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            expected,
            "unexpected hash for {}",
            path.display()
        );
        bytes
    }

    fn parse_ranges(value: &Value) -> Vec<(u32, u32)> {
        value
            .as_array()
            .unwrap_or_else(|| panic!("range list is not an array"))
            .iter()
            .map(|range| {
                let pair = range
                    .as_array()
                    .unwrap_or_else(|| panic!("range is not a pair"));
                let start = pair
                    .first()
                    .and_then(Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok())
                    .unwrap_or_else(|| panic!("range start is invalid"));
                let end = pair
                    .get(1)
                    .and_then(Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok())
                    .unwrap_or_else(|| panic!("range end is invalid"));
                assert!(start <= end, "range is inverted");
                (start, end)
            })
            .collect()
    }

    fn in_ranges(scalar: u32, ranges: &[(u32, u32)]) -> bool {
        let next = ranges.partition_point(|(start, _)| *start <= scalar);
        next > 0 && scalar <= ranges[next - 1].1
    }
}
