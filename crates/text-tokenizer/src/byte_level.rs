//! The reversible GPT-2/ByteLevel byte alphabet.

use std::sync::OnceLock;

/// Maps raw bytes to the Unicode alphabet used by BPE vocabulary symbols.
#[must_use]
pub(crate) fn encode_bytes(bytes: &[u8]) -> String {
    let table = tables();
    bytes
        .iter()
        .map(|byte| table.encode[usize::from(*byte)])
        .collect()
}

/// Reverses a vocabulary symbol to its raw bytes. A symbol containing a
/// character outside the `ByteLevel` alphabet is not decodable by this profile.
#[must_use]
pub(crate) fn decode_symbol(symbol: &str) -> Option<Vec<u8>> {
    let table = tables();
    symbol
        .chars()
        .map(|character| table.decode.get(&character).copied())
        .collect()
}

struct ByteTables {
    encode: [char; 256],
    decode: std::collections::HashMap<char, u8>,
}

fn tables() -> &'static ByteTables {
    static TABLES: OnceLock<ByteTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut selected = Vec::new();
        selected.extend(33_u8..=126);
        selected.extend(161_u8..=172);
        selected.extend(174_u8..=255);

        let mut encode = ['\0'; 256];
        let mut extra = 0_u32;
        for byte in 0_u8..=255 {
            if selected.contains(&byte) {
                encode[usize::from(byte)] = char::from(byte);
            } else {
                let character = char::from_u32(256 + extra).unwrap_or(char::REPLACEMENT_CHARACTER);
                encode[usize::from(byte)] = character;
                extra += 1;
            }
        }
        let mut decode = std::collections::HashMap::with_capacity(256);
        for byte in 0_u8..=255 {
            decode.insert(encode[usize::from(byte)], byte);
        }
        ByteTables { encode, decode }
    })
}

#[cfg(test)]
mod tests {
    use super::{decode_symbol, encode_bytes};

    #[test]
    fn all_bytes_round_trip_through_bytelevel_symbols() {
        let bytes: Vec<u8> = (0_u8..=255).collect();
        let mapped = encode_bytes(&bytes);
        assert_eq!(decode_symbol(&mapped).as_deref(), Some(bytes.as_slice()));
    }
}
