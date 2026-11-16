# Offline feedback page

The offline page accepts `ggtts-feedback-input/1` and exports
`ggtts-feedback-result/1`. The JSON schemas live in
`schemas/ggtts-feedback-input-1.schema.json` and
`schemas/ggtts-feedback-result-1.schema.json`. Both formats use stable,
nonempty ASCII IDs: a letter or digit followed by at most 127 letters, digits,
dots, underscores or hyphens. Input item IDs must be unique.

An input has `format`, `id`, and a nonempty `items` array. Each item has
`id`, nonempty Unicode scalar `text`, and `audio`, a
`data:audio/flac;base64,...` URI. The generator checks base64 and the FLAC
magic; it does not decode the audio or certify its integrity.

A result has `format`, the same page `id`, a page `note`, and one item
for every input item. Each result item has `id`, `marks`, `rating`
(`accept`, `reject`, or `unsure`) and `note`. Marks are unique,
zero-based indices into the text's CJK scalar values, not UTF-16 offsets or
indices into all characters. The ranges are U+3400–4DBF, U+4E00–9FFF,
U+F900–FAFF, U+20000–2FA1F and U+30000–323AF. Punctuation, Latin text and
emoji do not occupy an index. For `A中😀𠀀，文`, the valid indices are
0, 1 and 2. Rating is independent of marks.

Generate and validate with Python 3:

```sh
python3 scripts/feedback_page.py build input.json page.html
python3 scripts/feedback_page.py check-result input.json result.json
python3 -m unittest scripts.test_feedback_page
```

The build refuses an existing destination. It inlines the supplied audio and
escaped JSON in the local template; input strings cannot terminate the JSON
script element. Schema validation alone cannot compare a result against its
input text, so use `check-result` to check identity, item completeness and
mark bounds before consuming an export. An invalid result exits nonzero.
Use synthetic data when testing; the page contains the supplied text and audio
and should be handled as an artifact containing those inputs.
