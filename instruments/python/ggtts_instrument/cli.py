"""The command line the runner passes an event instrument, and `run()`, which keeps a crash from going unreported.

The runner (ggtts#30) starts each instrument after the render as:

    PROGRAM [--events E.jsonl] --audio OUT.flac --envelope ENV.json --name NAME [--KEY VALUE]...

`--events` is the render's trace, `-` for standard input. Every `--KEY VALUE` the runner does not define is the
instrument's own configuration, and it lands in the envelope's `provenance.config` as given. Three more are read
here when present: `--run` (the run id), `--instrument` (this instance's id) and `--render` (the render's id). Each
defaults to something stable, and the defaults are recorded in the configuration, so no id is invented silently.
"""

import os
import sys
import traceback

from .envelope import Envelope, EnvelopeError
from .audio import _start_decoder_capture, _stop_decoder_capture

_OWN = ("events", "audio", "envelope", "name", "run", "instrument", "render")


class Context:
    """What one invocation was given."""

    def __init__(self, events, audio, envelope, name, run, instrument, render, config):
        self.events = events
        self.audio = audio
        self.envelope = envelope
        self.name = name
        self.run = run
        self.instrument = instrument
        self.render = render
        self.config = config

    def number(self, key, default):
        """A numeric setting from the configuration, recorded with its default when absent."""
        v = self.config.get(key)
        if v is None:
            self.config[key] = default
            return default
        try:
            x = float(v)
        except ValueError:
            raise SystemExit("--%s must be a number, not %r" % (key, v))
        self.config[key] = x
        return x


def parse_args(argv=None, default_name=None):
    """Parse `--KEY VALUE` pairs. Refuses a flag without a value, and a missing `--envelope`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    got = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("--") or len(a) == 2:
            raise SystemExit("expected --KEY VALUE, got %r" % a)
        if "=" in a:
            k, v = a[2:].split("=", 1)
            i += 1
        else:
            if i + 1 >= len(argv):
                raise SystemExit("%s needs a value" % a)
            k, v = a[2:], argv[i + 1]
            i += 2
        got[k.replace("-", "_")] = v
    if "envelope" not in got:
        raise SystemExit("--envelope PATH is required")
    config = {k: v for k, v in got.items() if k not in _OWN}
    name = got.get("name") or default_name
    if not name:
        raise SystemExit("--name NAME is required")
    audio = got.get("audio")
    render = got.get("render")
    if render is None:
        render = os.path.splitext(os.path.basename(audio))[0] if audio else "render"
        config["render_default"] = render
    run = got.get("run")
    if run is None:
        run = "local"
        config["run_default"] = run
    return Context(got.get("events"), audio, got["envelope"], name, run, got.get("instrument") or name, render,
                   config)


def run(measure, name, version, hooks, audio_state="n/a", argv=None):
    """Parse the arguments, call `measure(ctx, env)`, and write the envelope.

    `measure` fills `env` (an `Envelope`) and may set its status. If it raises, the envelope is written `failed`
    with the exception's last line in `error` and the traceback on standard error, so the runner's log has it.
    Returns 0 for a successfully produced `ok`, `skipped`, or `truncated` envelope, so the runner can retain
    its reason and partial results. A failed instrument still exits 1.
    """
    ctx = parse_args(argv, default_name=name)
    env = Envelope(ctx.name, version, ctx.run, ctx.instrument, hooks, ctx.config, audio_state)
    decoders, token = _start_decoder_capture()
    try:
        measure(ctx, env)
    except Exception as e:  # noqa: BLE001 - any failure of the instrument is reported, never swallowed
        traceback.print_exc()
        env.doc["metrics"] = []
        env.status("failed", "%s: %s" % (type(e).__name__, e))
    finally:
        _stop_decoder_capture(token)
    env.doc["provenance"]["config"] = dict(ctx.config)
    env.doc["provenance"]["config"]["ggtts_audio_decoders"] = decoders
    for decoder in decoders:
        if decoder["sha256"]:
            marker = {"path": "decoder:" + decoder["kind"], "sha256": decoder["sha256"]}
            if marker not in env.doc["provenance"]["inputs"]:
                env.doc["provenance"]["inputs"].append(marker)
    try:
        env.write(ctx.envelope)
    except EnvelopeError as e:
        sys.stderr.write("the instrument built an invalid envelope: %s\n" % e)
        env.doc["metrics"], env.doc["artifacts"] = [], []
        env.status("failed", "invalid envelope: %s" % e)
        env.write(ctx.envelope)
    return 0 if env.doc["status"] in ("ok", "skipped", "truncated") else 1

