# Adding a capability

A capability is an option a render takes: a style instruction, steering, a seed salt, a trace, an edge kept at each
chunk. In ggtts it is defined once, as a row of the options table, and every front end either passes it through or
refuses it by name ([architecture-direction.md](architecture-direction.md)). This page walks one option from the row
to the conformance test. Each step names the file it touches; a new option touches these files and no others, or
the pull request says why.

The worked example is `load.kitten_full_head` (`--kitten-full-head`, `GGTTS_KITTEN_FULL_HEAD`), added in
[#296](https://github.com/dandelion-love/ggtts/issues/296) (PR [385](https://github.com/dandelion-love/ggtts/pull/385)):
an engine-specific switch with an environment spelling, a capability only one engine has, and a field the record
carries.

## 1. The row: `crates/ggtts-core/src/options.rs`

Add one `Descriptor` to `TABLE`, in its section, with:

- the **phase**: `Load` when it is fixed when the engine opens, `Render` when it is chosen per render;
- the **kind** and **default**: the default is today's behaviour, so rendering with every default is unchanged;
- every **spelling** it has: `flags` for `ggtts-render` and `ggtts-book`, `env` when an environment variable sets it,
  and `python`, `c`, `kotlin` for a binding's own argument. An option reached only through an options document needs
  no binding spelling;
- one line of **help**.

Add its field to the section's struct (`Load`, `Render`, `Observe`, `Output` or `Book`) with the same name, and to the
section's `Default`. The parsers, the schema and the environment rule read the row, so nothing else in the module
needs it unless the option is recorded (step 4).

Then regenerate the schema and give a fixture the key:

- `schemas/ggtts-options-1.schema.json`: run `GGTTS_BLESS=1 cargo test -p ggtts-core --test options` and review
  the diff;
- `fixtures/options/<section>.json`: add the key, so `the_fixtures_give_every_option` sees it.

## 2. Which engines have it: `crates/ggtts-core/src/capabilities.rs`

When only some engines support the option, add a field to `Capabilities`, set it in `Capabilities::of` for the
engines that have it, and add the key to `lacking()` in `options.rs`. `options::check` then applies the one rule
everywhere: a value given by flag or document is refused with `Capabilities::refusal`'s wording, and a value that
came only from the environment is ignored with one warning and recorded as ignored. Give the key its own refusal
wording in `Capabilities::refusal` when the default (`the <engine> engine has no <key>`) says too little.
`capabilities::statement` destructures `Capabilities`, so a new field fails to compile until the C ABI's statement
carries it (`crates/ggtts-ffi/tests/introspection.rs` and the `docs/bindings/c.md` example follow).

An option every engine takes skips this step.

## 3. Where it is read

The option reaches the code that acts on it through the shared value, never through a front end's own argument:

- a **render** option through `Options::render_options` (`RenderOptions`) and the pipeline
  (`crates/ggtts-core/src/pipeline.rs`), which every front end renders through;
- a **load** option through `ggtts_load::Request` (which is `Load`) and the loaders: `crates/ggtts-load/src/lib.rs`
  for `ggtts-book` and `crates/ggtts-cli/src/bin/ggtts-render.rs`'s `load_engine` (the example sets
  `KittenEngine::full_head` in both);
- an **observe** or **output** option in the front end that writes the file or the stream.

## 4. The record: `options::record` and `REPORTED`

When the option changes the audio, `options::record` writes it and, when every report should carry it, `REPORTED`
names it. The trace header (`crates/ggtts-core/src/trace.rs`, `Header::set_reported`) and the timing and duration
reports then carry it, so their schemas gain the field (`schemas/ggtts-trace-2.schema.json`,
`schemas/ggtts-timing-1.schema.json`, `schemas/ggtts-duration-1.schema.json`), added within their format versions.
A book records it in each chapter's render identity (`crates/ggtts-book/src/render.rs`, `render_identity`), so a
resume under another value starts the chapter again.

## 5. Each front end carries or refuses it

- **`ggtts-render`**: it takes every table flag outside the book section. Add the flag to `USAGE`
  (`usage_names_every_table_flag_ggtts_render_takes_and_no_other` fails until it is there). When one mode cannot use
  it, refuse it there by name: `--text` refuses `--seed-salt` and `--edge-s`, for example.
- **`ggtts-book`**: add the key to `RENDER_TAKES` in `crates/ggtts-book/src/options.rs` when a book carries it, or
  give `refusal()` its reason when it does not.
- **The bindings**: an options document reaches `crates/ggtts-load/src/binding.rs`, whose `document_refusal` names
  every key a binding refuses and why. A binding argument of its own (a `python`, `c` or `kotlin` spelling) changes a
  surface snapshot (`conformance/surfaces.txt`), which must be updated in the same pull request.

## 6. The conformance table: `crates/ggtts-cli/tests/conformance.rs`

Give the option a cell on every front end in `FRONT_ENDS`: `Carried`, or `Refused` with the reason the front end
gives. A new option never enters as a `Gap`: `conformance/gaps.txt` only shrinks
(`scripts/check_conformance_base.py --base <ref>`). Then add a probe where the option can be seen: a file written, a
trace line, a seed recorded, a refusal by name. A cell without a probe is counted as declared, not exercised, and the
test prints how many there are. The example has probes on `ggtts-render` (the capability refusal on the fake
engine) and the C ABI (the document refusal).

## 7. Tests, docs and the CHANGELOG

- a test of the rule each way: carried where it is had, refused or ignored where it is not (the example adds two in
  `crates/ggtts-core/tests/capabilities.rs`);
- `docs/options.md` when the option has an environment spelling or changes the record;
- one CHANGELOG line under **Unreleased**.

A change to the options table also means running `crates/ggtts-py`'s own `cargo test` and the wheel's pytest: that
crate is outside the workspace, and `cargo test --workspace` does not build it.
