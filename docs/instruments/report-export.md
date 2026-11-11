# Export an instrument report

The runner's `report.json`, `report.md`, `report.html`, `run.log`, and each
instrument envelope, log and artifact are private. They may contain paths,
references, transcripts or secrets. This command creates a separate JSON file;
it never overwrites a destination or modifies the original. Publication uses a
same-directory hard link and refuses if that operation is unavailable. An
interrupted process may leave a temporary file containing only the sanitized
output; remove it if present:

```sh
cargo run -p ggtts-instruments --bin ggtts-report-export -- private/report.json shareable.json
```

Only `ggtts-report/2` merged JSON is accepted. The exporter validates each
envelope, rejects unknown fields and inconsistent completeness or counts, and
refuses unsupported metrics. The reviewed name/unit pairs in `METRICS` cover
the bundled catalogue's metrics, including ASR, audio levels, timing, pauses,
pitch and loudness. Unreviewed third-party metrics are refused. Extend the
allowlist and output schema together only after reviewing the new name and unit
as public constants. Failure prints a generic message; it does
not print input contents or paths. Do not treat a failed export as a report.

Output retains status, required disposition, metric values and direction,
trim state, input/model/artifact SHA-256 values, and the input report's
instrument order. Instrument names become ordinal identifiers. Run, instance
and render identifiers, paths, text, free-form config, versions, hooks, logs,
errors and artifact types are omitted. Render scopes become a boolean, not a
joinable identifier; absent details must be recovered from private evidence.
Hashes identify content, but can be matched against known inputs; metric values
themselves may be sensitive. Review the output in its intended context before
sharing. The export is a separate format, not a replacement for the private
report, and does not establish ASR quality or detect omissions. The output
schema is `schemas/ggtts-shareable-report-1.schema.json`; schema validation
alone is not an authorization to share the data.
