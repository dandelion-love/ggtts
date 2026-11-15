# This repo's board layer

Read after `.gtd/kit/BOARD.md`. Where this file differs from the kit, this file wins, except that it can never relax
the kit's rule 1 (board only through the gtd MCP server) or rule 9 (hard limits), apart from the grants rule 9 allows
below. It lives in this repo, not in the public kit, so it is the place for anything specific to ggtts.

**This repository is public.** Everything an agent writes here (issues, comments, pull requests, commits, code, docs,
fixtures, release notes) is world-readable. Write in ggtts's own terms: its issues, crates, scripts and published
models. Never name or link a private repository, its issues, runs, experiments, hosts, paths or buckets, and never
copy text from one. A private project that depends on ggtts links to ggtts issues from its own side; ggtts never
links back.

## Board and profile

- Profile: `software` (as in `.github/gtd-context.json`)
- Board: `dandelion-love/4`
- Other boards or channels agents here may write to: none
- Knowledge bundle: none yet. Design and user docs live in `docs/`; the direction page is
  `docs/architecture-direction.md`.
- The owner's `needs` value: `needs:james`
- Standing chore item: none yet
- MCP server: reach it through the gtd connector you already have. Its address is not kept in this public repo; ask
  the maintainer if you have no connector.

## Notifications

- Pinged for asks: @hobpond (`notify.mentions` in `.github/gtd-context.json`)
- Louder channel: web-push
- New issues: view

## Labels this repo adds or uses differently

| Label | Meaning here | What an agent does with it |
| --- | --- | --- |
| agent-proposed | filed by an agent, or a request from another project | waits for the maintainer or a review that promotes it; never built unpointed |
| agent-review | asks another agent session to review the issue or its pull request | review it with `board-review`, from a session that did not work it |

The full list is the `software` profile's (`profile_get`).

## Status per label (only if this repo has no label-to-Status bridge)

This repo has no `project-sync.yml`, so an agent that sets one of these labels also sets Status with
`project_set_status`, as kit rule 3 says.

| Label | Status column |
| --- | --- |
| agent-running | In Progress |
| waiting-for-review | In Review |
| blocked | Blocked |
| agent-failed | Blocked |
| needs:james | Needs owner |
| agent-proposed | Backlog |

Closed items: Done, set by the server's bot on close. See `.gtd/kit/knowledge/model/board-fields.md`, "Closed is
Done".

## Fields and columns

The kit's standard fields. An agent sets Size on an unsized item it builds, from the acceptance criteria, and says why
in one line on the item. Work that is really L or XL is split, never sized down.

## Queue and ETA

- Queue view: Agent queue
- ETA field: Target date
- Start field: Start date

## What agents may pick up here

The kit's rule: unpointed, only `agent-ready` items of Size XS to M with no `agent:<x>` label naming someone else.
Pointed by the maintainer, any size.

## Procedure differences

### Integration branch `rc`

Pull requests target `rc`, not `main`. `rc` is where parallel work meets and is tested on CI. `main` moves only by
promoting `rc` (see the merge rules below). `CONTRIBUTING.md` describes the same flow for contributors.

### Agent pull requests open non-draft

Open every agent pull request with `pr_open` and `draft: false`, and mark work in progress in the title. A draft holds a
reviewed, green pull request until a person clicks.

### Merges into `rc` on a review and green CI; revert, don't wait

The maintainer's standing direction for this repo, recorded by the pull request that added this section: agents do not
wait on the maintainer's review to merge into `rc`, because a bad merge is cheap to revert.

- **Merging into `rc`.** An agent merges a pull request into `rc` once all three hold:
  - another agent session (never the one that did the work) has reviewed it with `board-review` and found it ready;
  - every CI check on the head is green, read just before merging;
  - the pull request changes no workflow beyond the `ci` grant below.
- **Promoting `rc` to `main`.** An agent promotes `rc` to `main` the same way, after `scripts/rc_gate.sh` passes on `rc`,
  and attaches its `rc-gate.json`.
- **Release tags stay the maintainer's.** An `-rc.N` pre-release tag is allowed.
- **A bad merge is reverted, never force-pushed.** The revert pull request follows the same rule. Every merge is reported
  on its issue with its commit, so a revert is one step.
- **This relaxes kit rule 9**, which allows a merge only under a grant naming every file. The kit has no pattern grant
  yet, and a list of exact files cannot cover code work. Until the kit can express this, this section is the
  maintainer's explicit exception, and it lapses if the maintainer removes it. Pull requests that change `.gtd/` or
  `.github/gtd-context.json` are never merged by an agent.

### Deferred work is filed with its decision

A decision, review or resolution that defers part of the work ("later", "a follow-up") captures that story in the same
turn, with an outcome and acceptance criteria, and links it.

### A capability goes on one shared surface

A capability (an option, a mode, an instrument) is defined once, as data with a schema, on the shared render options
value (epic #263). Front ends pass it through, and a conformance test fails when a front end neither carries it nor
refuses it. A flag or code path that exposes a capability on one front end only is a review finding. See
`docs/architecture-direction.md`.

## Standing lanes

None in this repo. Agents arrive pointed by the maintainer, or pick `agent-ready` items under the rule above.

## Intake

Other projects file requests here as issues labelled `agent-proposed`, written in ggtts's terms as a public user would
meet the problem. A request stays `agent-proposed` until the maintainer, or a review that finds it ready, promotes it.

- label: agent-proposed
- ack_within: 24h
- owner_needs: james

## Workflow grants

### grant: ci

- paths:
  - .github/workflows/ci.yml
  - .github/workflows/release.yml
- granted_by: hobpond
- date: 2026-11-05
- link: the pull request that added this section (granted 2026-11-05 and widened 2026-11-06, carried over here)
- also: any change to these two files' jobs, steps and triggers on `push`, `pull_request` and tags. It never widens
  `permissions:` (contents: read in ci.yml, and release.yml's existing release and attestation permissions), and never
  adds a secret, `pull_request_target`, `workflow_run`, a self-hosted runner, or an action not pinned to a full commit
  SHA. A pull request touching these files may merge under the `rc` rule above.

## Merge grants

None by file. Merges follow "Merges into `rc`" above.
