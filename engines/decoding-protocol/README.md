# Two-stage decoding protocol

This separate Rust crate implements the draft-5 JSON, schema, framing, and argument teacher-trace rules. It retains earlier immutable protocol fixtures for compatibility checks.

Safe JSON serializes typed semantic values with RFC 8785 ordering and binary64 numeric rules, then escapes literal UTF-8 angle brackets. Raw JSON validation preserves source property order and numeric spelling where the protocol requires it. Payload segments are tokenized independently; reserved marker IDs are rejected outside explicit framing.

The schema implementation handles supported assertions, references, effective branches, finite values, presence choices, typed arrays, and atomic dynamic containers. Teacher planning validates the original instance before compiling argument traces.

The crate accepts an injected segment tokenizer. It does not load a language model or execute tools. Public route candidates, forced-operation records, globally interleaved operation order, and global probe indices remain pending work. Available-field corpus parity does not qualify those missing interfaces.

Run its standalone synthetic checks from the workspace:

    cargo +1.89.0 test -p minifield-decoding-protocol --locked
    cargo +1.89.0 clippy -p minifield-decoding-protocol --all-targets --locked -- -D warnings

The opt-in corpus bridge reads explicit external input paths. Private corpus rows and bridge output do not belong in this repository.
