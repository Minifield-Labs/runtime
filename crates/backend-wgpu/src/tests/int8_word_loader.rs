//! Portable contracts for the finite INT8 loader composition. These tests don't
//! prove compiled GPU arithmetic, scheduling, memory-bank behavior or speed.

use crate::{Nf4Staging, kernels::Kernel};
use std::collections::BTreeMap;

const STAGING: [Nf4Staging; 4] = [
    Nf4Staging::F32,
    Nf4Staging::F16Weights,
    Nf4Staging::F16Activations,
    Nf4Staging::F16Both,
];
const INT8: [Kernel; 4] = [
    Kernel::PackedGemmInt8,
    Kernel::PackedGemmPairInt8,
    Kernel::PackedSwigluGemmInt8,
    Kernel::PackedGemmPairSwigluInt8,
];

macro_rules! frozen {
    ($name:literal) => {
        include_str!(concat!(
            "../../tests/fixtures/packed-scalar-415007e/",
            $name
        ))
    };
}

#[test]
fn non_int8_compositions_keep_the_independent_baseline_bytes() {
    let forms = [
        (
            Kernel::PackedGemmNf4,
            frozen!("nf4_linear_header.wgsl"),
            true,
        ),
        (
            Kernel::PackedGemmTernary,
            frozen!("ternary_linear_header.wgsl"),
            false,
        ),
        (
            Kernel::PackedGemmPairNf4,
            frozen!("nf4_pair_header.wgsl"),
            true,
        ),
        (
            Kernel::PackedGemmPairTernary,
            frozen!("ternary_pair_header.wgsl"),
            false,
        ),
        (
            Kernel::PackedSwigluGemmNf4,
            frozen!("nf4_swiglu_header.wgsl"),
            true,
        ),
        (
            Kernel::PackedSwigluGemmTernary,
            frozen!("ternary_swiglu_header.wgsl"),
            false,
        ),
        (
            Kernel::PackedGemmPairSwigluNf4,
            frozen!("nf4_pair_swiglu_header.wgsl"),
            true,
        ),
        (
            Kernel::PackedGemmPairSwiglu,
            frozen!("ternary_pair_swiglu_header.wgsl"),
            false,
        ),
    ];
    for (kernel, header, nf4) in forms {
        for staging in STAGING {
            // Independent fixture selection, including the original F16 body.
            let (enable, tile) = match staging {
                Nf4Staging::F32 => ("", frozen!("nf4_prefill.wgsl").to_owned()),
                Nf4Staging::F16Weights => (
                    "enable f16;\n",
                    frozen!("nf4_prefill_f16.wgsl")
                        .replace("__STGX__", "f32")
                        .replace("__STGW__", "f16"),
                ),
                Nf4Staging::F16Activations => (
                    "enable f16;\n",
                    frozen!("nf4_prefill_f16.wgsl")
                        .replace("__STGX__", "f16")
                        .replace("__STGW__", "f32"),
                ),
                Nf4Staging::F16Both => (
                    "enable f16;\n",
                    frozen!("nf4_prefill_f16.wgsl")
                        .replace("__STGX__", "f16")
                        .replace("__STGW__", "f16"),
                ),
            };
            let expected = [
                enable,
                frozen!("wgsl_index.wgsl"),
                if nf4 { frozen!("nf4_lut.wgsl") } else { "" },
                header,
                &tile,
            ]
            .concat();
            assert_eq!(
                kernel.source(staging).as_bytes(),
                expected.as_bytes(),
                "{} {staging:?}",
                kernel.name()
            );
        }
    }
    let restored = include_str!("../shaders/packed_prefill.wgsl").replace(
        "__WEIGHT_TILE_LOADER__\n",
        include_str!("../shaders/packed_tile_scalar_loader.wgsl"),
    );
    assert_eq!(restored, frozen!("nf4_prefill.wgsl"));
}

#[test]
fn int8_keeps_geometry_barriers_bindings_and_f32_staging() {
    let bindings: [&[&str]; 4] = [
        &["dst", "x", "codes", "scales", "x4"],
        &[
            "dst_a", "dst_b", "x", "codes_a", "scales_a", "codes_b", "scales_b", "x4",
        ],
        &["dst", "gate", "up", "codes", "scales", "gate4", "up4"],
        &[
            "dst", "x", "codes_a", "scales_a", "codes_b", "scales_b", "x4",
        ],
    ];
    for (index, kernel) in INT8.into_iter().enumerate() {
        assert_eq!(kernel.storage_bindings(), [5, 8, 7, 7][index]);
        assert_eq!(
            kernel.read_only_mask(),
            [0b11_1100, 0b1_1111_1000, 0b1111_1100, 0b1111_1100][index]
        );
        let baseline = kernel.source(Nf4Staging::F32);
        for staging in STAGING {
            let source = kernel.source(staging);
            assert_eq!(source, baseline);
            assert!(!source.contains("__WEIGHT_TILE_LOADER__"));
            assert!(!source.contains("enable f16"));
            assert_eq!(source.matches("workgroupBarrier()").count(), 2);
            let input_store = source
                .find("inputs[r * 16u + x] = xv;")
                .expect("input staging");
            let active = source.find("if linear < 128u").expect("word ownership");
            let first_barrier = source.find("workgroupBarrier()").expect("barrier");
            assert!(input_store < active && active < first_barrier);
            let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
                panic!("{}: {}", kernel.name(), error.emit_to_string(&source))
            });
            assert_eq!(module.entry_points.len(), 1);
            assert_eq!(module.entry_points[0].workgroup_size, [16, 16, 1]);
            let mut actual = BTreeMap::new();
            for (_, global) in module.global_variables.iter() {
                if let Some(binding) = &global.binding {
                    assert_eq!(binding.group, 0);
                    actual.insert(
                        binding.binding,
                        (global.name.as_deref().expect("name"), global.space),
                    );
                }
            }
            assert_eq!(actual.len(), bindings[index].len() + 1);
            for (slot, &name) in bindings[index].iter().enumerate() {
                let key = u32::try_from(slot + 1).expect("binding fits");
                let (actual_name, space) = actual[&key];
                assert_eq!(actual_name, name);
                let writes = slot == 0 || (index == 1 && slot == 1);
                let access = if writes {
                    naga::StorageAccess::LOAD | naga::StorageAccess::STORE
                } else {
                    naga::StorageAccess::LOAD
                };
                assert_eq!(space, naga::AddressSpace::Storage { access });
            }
            assert_eq!(actual[&0].1, naga::AddressSpace::Uniform);
        }
    }
    let loader = include_str!("../shaders/packed_tile_int8_word_loader.wgsl");
    for required in [
        "let linear = x + 16u * y;",
        "let c = linear / 4u;",
        "let q = 4u * (linear % 4u);",
        "col0 + c < n && base + q + 3u < k",
        "weights_a[(q + j) * 32u + c] = wa[j];",
        "weights_b[(q + j) * 32u + c] = wb[j];",
    ] {
        assert!(loader.contains(required), "missing contract: {required}");
    }
    assert!(!loader.contains("return"));
    for header in [
        include_str!("../shaders/int8_linear_header.wgsl"),
        include_str!("../shaders/int8_pair_header.wgsl"),
        include_str!("../shaders/int8_swiglu_header.wgsl"),
        include_str!("../shaders/int8_pair_swiglu_header.wgsl"),
    ] {
        assert!(header.contains("row * (k / 4u) + col / 4u"));
        assert!(header.contains("row * (k / 128u) + col / 128u"));
        assert!(!header.contains("fn weight_a("));
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Provenance {
    stream: usize,
    byte: usize,
    scale: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mutation {
    None,
    RowAlias,
    ScaleShift,
    DuplicateStore,
    MaskedRead,
}

struct Tile {
    cells: Vec<Option<Provenance>>,
    writes: Vec<usize>,
    reads: BTreeMap<(usize, usize, usize), usize>,
}

fn candidate_tile(
    n: usize,
    k: usize,
    col0: usize,
    base: usize,
    streams: usize,
    mutation: Mutation,
) -> Tile {
    let mut tile = Tile {
        cells: vec![None; streams * 512],
        writes: vec![0; streams * 512],
        reads: BTreeMap::new(),
    };
    for y in 0..16 {
        for x in 0..16 {
            let linear = x + 16 * y;
            if linear >= 128 {
                continue;
            }
            let column = linear / 4;
            let quartet_start = 4 * (linear % 4);
            let valid = col0 + column < n && base + quartet_start + 3 < k;
            for stream in 0..streams {
                let row = if mutation == Mutation::RowAlias {
                    (col0 + column) % 32
                } else {
                    col0 + column
                };
                let word = row * (k / 4) + (base + quartet_start) / 4;
                let scale = row * (k / 128)
                    + (base + quartet_start) / 128
                    + usize::from(mutation == Mutation::ScaleShift);
                if valid || mutation == Mutation::MaskedRead {
                    *tile.reads.entry((stream, word, scale)).or_default() += 1;
                }
                for j in 0..4 {
                    let reduction = if mutation == Mutation::DuplicateStore {
                        quartet_start
                    } else {
                        quartet_start + j
                    };
                    let slot = stream * 512 + reduction * 32 + column;
                    tile.writes[slot] += 1;
                    tile.cells[slot] = valid.then_some(Provenance {
                        stream,
                        byte: word * 4 + j,
                        scale,
                    });
                }
            }
        }
    }
    tile
}

// The former loader uses y as K coordinate and x/lo as column. It doesn't
// call the candidate's quartet or address helper.
fn scalar_tile(n: usize, k: usize, col0: usize, base: usize, streams: usize) -> Tile {
    let mut tile = Tile {
        cells: vec![None; streams * 512],
        writes: vec![0; streams * 512],
        reads: BTreeMap::new(),
    };
    for y in 0..16 {
        for x in 0..16 {
            for lo in 0..2 {
                let row = col0 + x + lo * 16;
                let byte = row * k + base + y;
                let scale = byte / 128;
                for stream in 0..streams {
                    let slot = stream * 512 + y * 32 + x + lo * 16;
                    tile.writes[slot] += 1;
                    if row < n && base + y < k {
                        tile.cells[slot] = Some(Provenance {
                            stream,
                            byte,
                            scale,
                        });
                        *tile.reads.entry((stream, byte / 4, scale)).or_default() += 1;
                    }
                }
            }
        }
    }
    tile
}

fn matches_direct_tensor(
    tile: &Tile,
    n: usize,
    k: usize,
    col0: usize,
    base: usize,
    streams: usize,
) -> bool {
    for stream in 0..streams {
        for d in 0..16 {
            for c in 0..32 {
                let row = col0 + c;
                let logical_byte = row * k + base + d;
                let expected = (row < n && base + d < k).then_some(Provenance {
                    stream,
                    byte: logical_byte,
                    scale: row * (k / 128) + (base + d) / 128,
                });
                let slot = stream * 512 + d * 32 + c;
                if tile.cells[slot] != expected || tile.writes[slot] != 1 {
                    return false;
                }
            }
        }
    }
    let expected_reads = streams * n.saturating_sub(col0).min(32) * 4;
    tile.reads.values().sum::<usize>() == expected_reads
        && tile
            .reads
            .keys()
            .all(|&(_, word, scale)| word < n * (k / 4) && scale < n * (k / 128))
}

fn check_input_stores_and_consumers(m: usize, k: usize, row0: usize, base: usize) {
    let mut inputs = [usize::MAX; 1024];
    let mut writes = [0; 1024];
    // All 256 invocations participate even when y >= 8.
    for y in 0..16 {
        for x in 0..16 {
            for hi in 0..4 {
                let t = y + hi * 16;
                let slot = t * 16 + x;
                writes[slot] += 1;
                if row0 + t < m {
                    inputs[slot] = (row0 + t) * k + base + x;
                }
            }
        }
    }
    assert!(writes.iter().all(|&count| count == 1));
    // Read ACTUAL flat shared input stores at unchanged fragment coordinates.
    for y in 0..16 {
        for r in 0..4 {
            let t = y + r * 16;
            for d in 0..16 {
                let expected = if row0 + t < m {
                    (row0 + t) * k + base + d
                } else {
                    usize::MAX
                };
                assert_eq!(inputs[t * 16 + d], expected);
            }
        }
    }
}

fn check_weight_consumers(
    tile: &Tile,
    n: usize,
    k: usize,
    col0: usize,
    base: usize,
    streams: usize,
) {
    // Read ACTUAL candidate stores, rather than computing values that bypass them.
    for x in 0..16 {
        for d in 0..16 {
            for c in [x, x + 16] {
                for stream in 0..streams {
                    let actual = tile.cells[stream * 512 + d * 32 + c];
                    let expected = (col0 + c < n).then_some(Provenance {
                        stream,
                        byte: (col0 + c) * k + base + d,
                        scale: (col0 + c) * (k / 128) + (base + d) / 128,
                    });
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

#[test]
fn independent_scalar_provenance_matches_actual_shared_consumers() {
    for m in [1_usize, 2, 4, 8, 16, 63, 64, 65] {
        for n in [1_usize, 17, 31, 32, 33] {
            for k in [128_usize, 256, 384] {
                for form in 0..4 {
                    let streams = if form == 1 || form == 3 { 2 } else { 1 };
                    for row0 in (0..m).step_by(64) {
                        for col0 in (0..n).step_by(32) {
                            for base in (0..k).step_by(16) {
                                let candidate =
                                    candidate_tile(n, k, col0, base, streams, Mutation::None);
                                let scalar = scalar_tile(n, k, col0, base, streams);
                                assert!(matches_direct_tensor(
                                    &candidate, n, k, col0, base, streams
                                ));
                                assert_eq!(candidate.cells, scalar.cells);
                                for (read, count) in &candidate.reads {
                                    assert_eq!(*count, 1);
                                    assert_eq!(scalar.reads[read], 4);
                                }
                                check_input_stores_and_consumers(m, k, row0, base);
                                check_weight_consumers(&candidate, n, k, col0, base, streams);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn provenance_oracle_rejects_aliases_missing_stores_and_masked_reads() {
    for mutation in [
        Mutation::RowAlias,
        Mutation::ScaleShift,
        Mutation::DuplicateStore,
        Mutation::MaskedRead,
    ] {
        let tile = candidate_tile(33, 256, 32, 128, 2, mutation);
        assert!(
            !matches_direct_tensor(&tile, 33, 256, 32, 128, 2),
            "{mutation:?}"
        );
    }
    // Masked reads are rejected even if every padded shared value is still zero.
    let masked = candidate_tile(17, 128, 0, 0, 1, Mutation::MaskedRead);
    let clean = candidate_tile(17, 128, 0, 0, 1, Mutation::None);
    assert_eq!(masked.cells, clean.cells);
    assert_ne!(masked.reads, clean.reads);
    let mut aliased_b = candidate_tile(33, 256, 32, 128, 2, Mutation::None);
    for value in aliased_b.cells.iter_mut().skip(512).flatten() {
        value.stream = 0;
    }
    assert!(!matches_direct_tensor(&aliased_b, 33, 256, 32, 128, 2));
    let mut missing = candidate_tile(33, 256, 32, 128, 2, Mutation::None);
    missing.writes[0] = 0;
    assert!(!matches_direct_tensor(&missing, 33, 256, 32, 128, 2));
}

#[test]
fn all_signed_bytes_and_non_power_scales_decode_before_product() {
    // Includes raw 0x80. The separate model-bundle loader still rejects -128;
    // an operation-level decoder must implement all 256 byte values correctly.
    for raw in 0_u16..=255 {
        let byte = u8::try_from(raw).expect("byte");
        for lane in 0..4 {
            let mut bytes = [1_u8, 129, 127, 255];
            bytes[lane] = byte;
            let word = u32::from_le_bytes(bytes);
            let extracted = u8::try_from((word >> (lane * 8)) & 255).expect("byte");
            let signed = i16::from(extracted) - if extracted >= 128 { 256 } else { 0 };
            assert_eq!(signed, i16::from(i8::from_ne_bytes([byte])));
            if byte >= 128 {
                assert_ne!(signed, i16::from(byte), "unsigned mutation");
            }
            for scale in [0.125_122_07_f32, 0.500_488_3, 0.063_537_6] {
                assert_eq!(
                    (f32::from(signed) * scale).to_bits(),
                    (f32::from(i8::from_ne_bytes([byte])) * scale).to_bits()
                );
            }
        }
    }
    let decoder = include_str!("../shaders/int8_decode_word.wgsl");
    assert!(decoder.contains("vec4<u32>(0u, 8u, 16u, 24u)"));
    assert!(decoder.contains("bitcast<vec4<i32>>(bytes << vec4<u32>(24u)) >> vec4<u32>(24u)"));
    assert!(decoder.contains("return vec4<f32>(signed) * scale;"));
}
