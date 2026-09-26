# Versioned contracts

Schemas and examples are checked locally, so this repository can work without sibling checkouts. contracts/lock.json records the owner, source/snapshot status, version, and SHA-256 of every schema.

- data-generation owns dataset and environment contracts.
- training owns the model-bundle contract.
- training also owns the optional quant-matrix diagnostic record. Its format description and tiny fixture are pinned under the lock file's artifacts field.
- Consumers vendor complete required schema sets under their versioned directories.
- Change the owner's source first, review compatibility, then explicitly copy the new version to consumers and refresh their lock files.
- Run contract checks in every affected repository. Preserve older versions while any supported artifact still uses them.
- Never quietly refresh schemas from the network at startup.

These snapshots describe artifact exchange. Passing their checks establishes structural compatibility; model execution and product acceptance require separate evidence.
