# Fleet board maintenance

This board reports operational observations. Paper claims continue to rely on
their registered source artifacts and project acceptance records.

The App coordinator owns this repository together with paper-progress. Local
and cloud operators may both develop, push PRs and merge verified work. Fetch
before work, keep branch changes scoped, and synchronize clean main checkouts
after merges. Preserve active execution directories at their registered SHAs.

The registered CaMCo R2 cache job reports copied/received bytes in MiB without
claiming GPU ownership. It checks the original process birth/boot identity,
fixed manifest and source commit, and terminal receipt/file metadata. Reaching
the byte total remains a running packaging/verification phase until a consistent
READY exists. The probe reads no model contents or scientific scores.

## Collection and publication

Configure an external cache using `FLEET_CACHE_DIR` or an ignored
`.fleet-local.json` based on the example. Windows/WSL task artifacts use physical
`D:\Codex` paths; the maintained host sets its cache under the task's `work`.
Place collector logs there as well. The existing ten-minute schedule can be
retained, pointing at `collect.py`.

The scheduled collector runs in Linux/WSL. Repository, PR and merge operations
may be initiated from either the local or cloud side.

The collector takes an exclusive local lock, synchronizes clean main, and probes
the configured machines read-only. A code update is picked up on the next
collection, without replacing a running probe. Snapshots are written outside
the checkout. `FLEET_PUSH=0` collects only and makes no Git commits or pushes.

Normal publication validates JSON structure, finite observations, unique job
identities, GPU ranges and append-only history. A separate Git index constructs
a commit containing only `data/fleet.json`, `data/curves.json` and
`data/history.jsonl`; it never stages the checkout. The publisher pushes a new
work branch, opens a PR, verifies the actual PR head/base/file scope, merges the
reviewed head through GitHub, confirms the merge, then fast-forwards local main.
It does not force-push, bypass checks, or include source/credential/config files
in an automatic data PR.

Conflicts, changed heads, failed checks and uncertain outcomes leave
`publication-pending.json` in the cache. Collection may continue; publication
waits for this App coordinator to inspect that specific branch/PR and reconcile
it. A later check may finalize an already-confirmed merge without sending
another merge request. Do not delete a pending record to force a retry. Successful publication
records live in `publications/pr-N.json`; the coordinator attaches every created
PR to its chat and records/synchronizes the result. Data branches stay until that
recording is complete. Other work and experimental branches are never cleaned
up by the collector.

Only one scheduled publisher should be active for a given cache/owner. Cloud
changes to the same snapshot files require checking the current publisher and
PR state first; neither side may overwrite diverged history.

## Verification

Run `python3 -m unittest test_active_jobs test_publishing` for status semantics
and publishing transaction checks. These tests use local fixtures and temporary
Git repositories; they do not run experiments or publish to GitHub. A real
collection with `FLEET_PUSH=0` can then verify the source observations in the
external cache before the first publication through the PR path.

For cache status changes, include `test_cache_range_status` in that test command.
