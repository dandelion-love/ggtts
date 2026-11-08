# Per-step stop-probability check

On one synthetic story, a real Qwen CPU render was traced and measured with `stop-probability` required. The
instrument emitted 86 records; each step index, codebook-0 token and stop weight matched the source trace, and one
terminal record matched the EOS token declared by that trace's producer. Removing the terminal step while leaving
the draw end and final trace marker caused the required measurement to exit 3 as incomplete. This is one Qwen
example, not proof for other engines or a quality judgment.

The instrument also ran from a packaged directory outside the checkout. The package includes its JSONL schema;
removing that schema caused the package smoke test to fail as intended.

The existing real-Qwen CPU no-change test also ran the full catalogue on its own fixture with this instrument
required; all ran successfully and the test observed unchanged full codec tokens, samples and output audio. The
fake-engine no-change test excludes this instrument because fake traces do not contain a separate terminal EOS
step. Neither result establishes GPU or cross-backend identity.

The synthetic story SHA-256 was `90e57e18382c982034cf5da44230689c89634576cb2daef2fb1d54b7679e5e3a`;
the talker and codec hashes were `33502b2035f97d7f458248d91e965a6800026549d382c5e8e300176027c70b67` and
`7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824`. The CPU executable SHA-256 was
`3254fa9ef2bc5424f9c7522d77f6104ce99751dd4c017a68fbb99512c071f5b8`, the resulting FLAC hash was
`915b19b5fc2ea864b6c90c12ef255c8109c7aa348d9e60e037289c39bb34f2a5`, and the per-step artifact hash was
`cef5e2fa7960703584461f6a1ff3bdea59cdaf9b75739c4b059f9418052cfc5f`. Those files remain external scratch data;
no text, audio or per-frame artifact is committed.

The producer's EOS token and probability-semantics markers are declarations, not independent validation of an
untrusted trace. Older unmarked traces and engines without this contract are incomplete when this instrument is
required. A same-input comparison with the lab's stop measure has not been established.
