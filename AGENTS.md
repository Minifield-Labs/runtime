# Repository instructions

This repository is part of Minifield Labs and must work as a standalone clone.

- Build product specialists that preserve supported behavior and reject unrelated requests. Track false rejection alongside task success.
- Keep code within this repository's responsibility, as described in README.md.
- Keep core inference in Rust, with browser integration in JavaScript. Platform owns product setup, durable jobs, artifact distribution, and releases; runtime owns local inference and evidence for the exact delivered bundle.
- Exchange versioned artifacts or the environment protocol. Never import a sibling repository by filesystem path or rely on a parent package manager.
- Keep private fixtures, expected outcomes, judge results, and lineage out of model-visible input. Preserve the application's authorization checks.
- Keep datasets, model weights, run output, customer data, and credentials out of Git. Only small synthetic contract fixtures belong in examples/.
- Treat schema versions as immutable once consumed. Update contract snapshots and their checksums explicitly, with compatibility checks.
- Read the local README before changing a module. Preserve other contributors' work.
- Use conventional commits. Don't publish remotes, start paid jobs, or collect customer data as part of scaffolding.
- Run `python scripts/check_contracts.py` after changing contracts or examples. This checks structure and fixture consistency; product behavior requires its own tests.
- Write concise documentation, use contractions naturally, and avoid em dashes.
