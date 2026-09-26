# Security

This is an experimental inference runtime. The supported development target is the current `main` branch; versioned stable-release support hasn't been established.

Treat model bundles and tokenizers as untrusted inputs. Format parsing should reject malformed dimensions, offsets, encodings, and numeric values before allocation or dispatch. Backend limits bound owned buffers and completion results; they don't bound the GPU driver's internal allocations or the entire host process.

For a suspected vulnerability, use the repository's private GitHub vulnerability-reporting channel if it's enabled. Include a small synthetic reproduction, revision, platform, and expected failure boundary. Keep credentials, customer data, and private model assets out of public reports.
