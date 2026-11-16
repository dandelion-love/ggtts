# Desktop feedback

A page produced by `ggtts-render --feedback-page` can be opened in the
system's browser without a server:

```sh
python3 scripts/feedback_desktop.py open read.html --audio read.flac
python3 scripts/feedback_desktop.py save read.html feedback.json --audio read.flac
```

Use the FLAC written with the page. The adjacent
`read.html.provenance.json` companion must be present. The host verifies the
page against the current trusted template, the embedded input digest, the
companion's chunk identities and hashes, and the selected audio's SHA-256 and
FLAC header before opening or saving. Saving validates every result and CJK
mark against the embedded input and writes `read.flac.feedback.json` beside
the audio, with a temporary sibling and no-clobber publication. An existing
result is never replaced. A failure is reported and no partial result is
published. Rename or move an earlier result explicitly before saving another.

These hashes check consistency, not authorship: someone able to rewrite a page,
companion and audio can construct new matching evidence. The host does not
independently decode clip PCM to prove its relationship to the output; that is
established by the render producer and its sample-identity tests. Keep artifacts
from a trusted render together.

The page stores a draft in browser local storage keyed by input digest. Storage
availability and persistence for `file:` URLs vary by browser and profile;
if the page reports that it cannot save a draft, export feedback and later
import it into the same page. The browser must support FLAC playback. The page
uses no network resources or privileged bridge; it contains the supplied text
and audio.
