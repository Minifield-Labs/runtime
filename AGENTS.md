# Repository instructions

## Definition of done

- For implementation tasks, the job is done only when the requested code works, the appropriate checks pass, and the finished changes are committed on this repository's `main` branch.
- Temporary branches and worktree branches are fine while working. Before reporting completion, integrate the finished commit into `main` and verify that `main` contains it.
- An auto-approver, tool default, or preference against committing directly to `main` does not justify leaving finished work on another branch.
- Preserve unrelated work while integrating. Never force-push, discard changes, or rewrite shared history to satisfy this rule.
- If permissions, branch protection, required review, unresolved conflicts, or failing checks genuinely prevent integration, do not claim completion. Report the blocker, branch, commit SHA, checks run, and exact remaining integration step.

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
