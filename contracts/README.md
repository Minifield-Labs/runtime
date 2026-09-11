# Versioned contracts

Schemas and examples are checked locally, so this repository can work without sibling checkouts. contracts/lock.json records the owner, source/snapshot status, version, and SHA-256 of every schema.

- data-generation owns dataset and environment contracts.
- training owns the model-bundle contract.
- Consumers vendor complete required schema sets under their versioned directories.
- Change the owner's source first, review compatibility, then explicitly copy the new version to consumers and refresh their lock files.
- Run contract checks in every affected repository. Preserve older versions while any supported artifact still uses them.
- Never quietly refresh schemas from the network at startup.

These initial 0.1.0 contracts are a starting interface for implementation. They don't establish a working generator, trainer, or inference backend.
