# Model-bundle contract 0.1.0

Owner: training. Runtime consumes a pinned snapshot. The manifest describes a self-contained, immutable delivery artifact.

A bundle records its product identity/version, exact base-model revision, training run and dataset digest, engine requirements, quantization format, context/action limits, decoding settings, and every asset's relative path, SHA-256 hash, and byte count.

Required asset roles: weights (one or more shards), tokenizer, chat_template, and product_contract. Optional roles cover adapters and runtime configuration. The tokenizer role can appear multiple times when its format requires several files.

Training owns packing and metadata. Runtime owns engine compatibility, resource checks, permissions, model-to-action execution, and device evidence. Product updates must agree with the bundle's declared contract version.

Schema validation can't establish that weights load, that an adapter was merged correctly, or that a quantization format is supported. Test the exact delivered assets on the reference device.

The example bundle uses purpose=contract_fixture, the fixture engine, and tiny text/JSON assets. It contains no trained model. A real inference loader must reject contract fixtures.

Treat shipped schema versions as immutable. Coordinate an explicit version bump and refreshed consumer snapshot when the shape changes.
