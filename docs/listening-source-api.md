# Listening source API

This page defines the versioned JSON API used by the Flutter listening client and an assigned-session source. The source authenticates listeners and returns temporary signed audio URLs; the client never receives object-storage credentials.

Production API requests use HTTPS. HTTP is allowed only for a loopback test source. Send the bearer token only to the configured API origin. Never attach it to a signed audio request or redirect target. Treat signed URLs as secrets: do not log them or include them in analytics, crash reports, or error messages.

## Authentication

`POST /v1/auth/token` accepts `{"username":"...","password":"..."}` and returns `{"access_token":"...","token_type":"Bearer","expires_at":"..."}`. The token is opaque. Send it as `Authorization: Bearer <access_token>` to source API routes. Keep it in memory only; do not persist passwords or tokens in settings, queued marks, or logs.

An expired token returns `401` with `error.code = "token_expired"`. Preserve queued marks and ask the listener to sign in again.

## Assigned sessions and audio

`GET /v1/listening-sessions` returns `{"sessions":[...]}`, scoped to the authenticated listener. Each session has a stable opaque `id`, `title`, and `clips`. Each clip has a stable opaque `id`, exact displayed `text`, `audio_url`, and `audio_url_expires_at`. Clients must not parse IDs.

Fetch `audio_url` anonymously. If it returns `403`, call `GET /v1/listening-sessions/{session_id}/clips/{clip_id}/audio-url` for `{"audio_url":"...","expires_at":"..."}`, then retry the audio request once. If refresh or retry fails, explain the playback failure and offer retry. Never send the source bearer token to an audio origin or redirect target.

Expiry values use RFC 3339 timestamps in UTC, for example `2026-11-16T12:30:00Z`. The client may refresh proactively; server `403` remains authoritative.

## Marks and verdicts

`POST /v1/listening-sessions/{session_id}/marks` submits one clip result with `clip_id`, a clip-level `verdict`, optional free `note`, and zero or more character `marks`. Each mark has a half-open `[start,end)` range in Unicode scalar values (code points) against the exact displayed clip text. Offsets are not UTF-8 byte offsets, UTF-16 code-unit offsets, or grapheme-cluster indexes. Require `0 <= start < end <=` the text's scalar-value count. A supplementary-plane emoji occupies one position; a combining sequence occupies one position per scalar value. Verdict values are `accept`, `reject`, or `unsure`. Mark notes are nullable free text; the clip-level verdict is saved even with no marks. Reject IDs outside the addressed session and ranges past the text.

The client creates an opaque `Idempotency-Key` per exact submission and reuses it unchanged on retries. First acceptance returns `200` with `{"accepted":true,"clip_id":"..."}`. A duplicate key with the same body returns the original success without adding a second mark set. The same key with a different body returns `409` (`idempotency_conflict`). Keep marks queued after transient network/server failures and retry the same key and body. The queue may persist, but credentials may not. After `401`, retain the queue and resume after sign-in.

## Errors

Errors use `{"error":{"code":"...","message":"...","retryable":false}}`. Messages are safe to display and must not contain signed URLs or bearer tokens. `retryable` is true for transient server failures. A `403` from a signed audio URL triggers refresh; a source API `403` means authorization failed and must not be retried automatically.

See [`schemas/ggtts-listening-source-1.schema.json`](../schemas/ggtts-listening-source-1.schema.json) for machine-readable shapes. The loopback fake is [`apps/flutter/test/support/listening_source_fake.dart`](../apps/flutter/test/support/listening_source_fake.dart).

## Shared-page mapping

The shared page marks CJK ordinal positions as pronunciation errors; each maps to a one-scalar source range with verdict reject. Item rating maps to the clip verdict. The source clip note carries a JSON object with format ggtts-shared-notes/1 and string fields page_note and item_note; both are preserved exactly. The mapper refuses any encoded note exceeding the schema limit of 4000 Unicode scalar values. The same page note is included in each item submission so each carries the complete context. The adapter derives the digest from validated canonical input bytes and checks the supplied source IDs, exact texts, and fetched-audio SHA-256 against every page item. FLAC validation checks STREAMINFO only and does not decode or certify the full audio stream.
