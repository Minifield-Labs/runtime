# Third-party notices

The project source is MIT licensed. Dependencies retain their own licenses, recorded in their packages and Cargo/Python lockfiles.

`tools/quant-reference` contains Q4 and MFQ8 reference code adapted from Wires (`unrvl-embdb`). Its original MIT notice is preserved in [LICENSE](tools/quant-reference/LICENSE). The [transfer audit](docs/wires-audit.md) and [source digests](docs/wires-provenance.json) identify the source snapshot and changes. This tool supports an existing training compatibility probe; it isn't an inference backend.

Model weights, tokenizers, datasets, and external oracle assets aren't included under this repository's license. Synthetic fixtures created by this repository are identified as such. A converter records source identity and hashes; that record doesn't grant rights to redistribute a source model.
