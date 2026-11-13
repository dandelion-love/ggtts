---
type: Decision
title: Android work lands on one integration branch
description: Every Android change goes into track/android by a reviewed PR, and track/android reaches rc as one PR when a usable slice is complete.
tags: [android, process]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T07:30:00-07:00 }
---
# Decision

All Android work (the build, the binding, the AAR, the demo app and the on-device measurements) lands on the
long-lived branch `track/android`, created from `rc` at `552669186c83`. Each change is a small PR into
`track/android`, reviewed by a session that did not write it. `track/android` merges into `rc` as one PR, once a
usable slice is complete.

# Why

The Android track moves faster than a release can absorb, and the owner wanted every step undoable without
touching `rc`, which was frozen for a release tag when the track began. A PR into `track/android` can be reverted
without affecting any release.

# Consequences

- A PR that builds on an unmerged Android PR still targets `track/android`. Until the earlier PR merges, its diff
  includes the earlier PR's commits.
- CI runs on every PR into `track/android` as on any PR. The specialized jobs are chosen by the paths a PR
  changes.

# Undo

Merge nothing more into `track/android`, and delete the branch. `rc` is unaffected until the one merge PR.
