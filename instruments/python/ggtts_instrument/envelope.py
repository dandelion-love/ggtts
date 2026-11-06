"""Build a `ggtts-report/1` envelope, check it, and write it.

The contract is `schemas/ggtts-report-1.schema.json` (ggtts#29). `check()` restates it here so an instrument refuses
to write an envelope the merger would mark `invalid`, and it names every reason, not only the first. The test suite
holds `check()` to the planted defects the schema refuses.

Two rules worth knowing:
- **A value the instrument could not compute is `None` (JSON null), never 0.**
- **Every metric says which way is better:** `higher`, `lower`, `neutral` for a descriptive value, or `n/a` for a
  diagnostic.
"""

import json
import os
import re

from .audio import sha256_of

FORMAT = "ggtts-report/1"
READS = (FORMAT,)
STATUSES = ("ok", "failed", "timeout", "skipped", "truncated", "invalid")
DIRECTIONS = ("higher", "lower", "neutral", "n/a")
AUDIO_STATES = ("raw", "trimmed", "n/a")

_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


class EnvelopeError(ValueError):
    """The envelope would not validate. `reasons` lists every one."""

    def __init__(self, reasons):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def _text(v):
    return isinstance(v, str) and v != ""


def _uint(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fields(obj, allowed, at, why):
    for k in obj:
        if k not in allowed:
            why.append("%s`%s` is not a field here" % (at, k))


def check(env):
    """Every reason `env` (a dict) is not a valid `ggtts-report/1` envelope; empty when it is."""
    why = []
    if not isinstance(env, dict):
        return ["an envelope is a JSON object"]
    _fields(env, ("$schema", "format", "name", "version", "identity", "provenance", "hooks", "status", "metrics",
                  "artifacts", "log", "error"), "", why)
    if env.get("format") not in READS:
        why.append("format %r is not one this writes (%s)" % (env.get("format"), ", ".join(READS)))
    if not (_text(env.get("name")) and _NAME.match(env["name"])):
        why.append("`name` must match %s" % _NAME.pattern)
    if not _text(env.get("version")):
        why.append("`version` must be a non-empty string")
    if env.get("status") not in STATUSES:
        why.append("status %r is not one of %s" % (env.get("status"), ", ".join(STATUSES)))
    hooks = env.get("hooks")
    if not isinstance(hooks, list) or not all(_text(h) for h in hooks):
        why.append("`hooks` must be a list of non-empty strings")
    if "log" not in env or not (env["log"] is None or _text(env["log"])):
        why.append("`log` must be a non-empty path or null")
    if env.get("error") is not None and not isinstance(env.get("error"), str):
        why.append("`error` must be a string or null")

    ident = env.get("identity")
    if not isinstance(ident, dict):
        why.append("`identity` must be an object {run, instrument}")
    else:
        _fields(ident, ("run", "instrument"), "identity: ", why)
        for k in ("run", "instrument"):
            if not _text(ident.get(k)):
                why.append("identity: `%s` must be a non-empty string" % k)

    prov = env.get("provenance")
    if not isinstance(prov, dict):
        why.append("`provenance` must be an object")
    else:
        _fields(prov, ("config", "ggtts", "engine", "backend", "model_sha256", "seeds", "inputs", "audio"),
                "provenance: ", why)
        if not isinstance(prov.get("config"), dict):
            why.append("provenance: `config` must be an object (the effective configuration)")
        for k in ("ggtts", "engine", "backend"):
            if k not in prov or not (prov[k] is None or _text(prov[k])):
                why.append("provenance: `%s` must be a non-empty string or null" % k)
        if "model_sha256" not in prov or not (prov["model_sha256"] is None or
                                              (_text(prov["model_sha256"]) and _SHA.match(prov["model_sha256"]))):
            why.append("provenance: `model_sha256` must be 64 lower-case hex digits or null")
        if prov.get("audio") not in AUDIO_STATES:
            why.append("provenance: `audio` must be one of %s" % ", ".join(AUDIO_STATES))
        seeds = prov.get("seeds")
        if not isinstance(seeds, list):
            why.append("provenance: `seeds` must be a list")
        else:
            for i, s in enumerate(seeds):
                at = "provenance: seeds[%d]: " % i
                if not isinstance(s, dict):
                    why.append(at + "must be an object")
                    continue
                _fields(s, ("render", "chunk", "seed"), at, why)
                if not _text(s.get("render")) or not _uint(s.get("seed")) or ("chunk" in s and not _uint(s["chunk"])):
                    why.append(at + "needs a non-empty `render`, a non-negative `seed`, and `chunk` if any >= 0")
        inputs = prov.get("inputs")
        if not isinstance(inputs, list):
            why.append("provenance: `inputs` must be a list")
        else:
            for i, x in enumerate(inputs):
                at = "provenance: inputs[%d]: " % i
                if not isinstance(x, dict):
                    why.append(at + "must be an object")
                    continue
                _fields(x, ("path", "sha256"), at, why)
                if not _text(x.get("path")) or not (_text(x.get("sha256")) and _SHA.match(x["sha256"])):
                    why.append(at + "needs a `path` and a lower-case hex `sha256`")

    metrics = env.get("metrics")
    if not isinstance(metrics, list):
        why.append("`metrics` must be a list")
    else:
        for i, m in enumerate(metrics):
            at = "metrics[%d]: " % i
            if not isinstance(m, dict):
                why.append(at + "must be an object")
                continue
            _fields(m, ("name", "value", "unit", "direction", "interval", "render", "chunk"), at, why)
            if not (_text(m.get("name")) and _NAME.match(m["name"])):
                why.append(at + "`name` must match %s" % _NAME.pattern)
            if "value" not in m or not (m["value"] is None or _number(m["value"])):
                why.append(at + "`value` must be a number, or null when it could not be computed")
            if not _text(m.get("unit")):
                why.append(at + "`unit` must be a non-empty string (\"1\" for a count or ratio)")
            if m.get("direction") not in DIRECTIONS:
                why.append(at + "`direction` must be one of %s" % ", ".join(DIRECTIONS))
            iv = m.get("interval")
            if "interval" in m and not (isinstance(iv, list) and len(iv) == 2 and all(_number(x) for x in iv)
                                        and iv[0] <= iv[1]):
                why.append(at + "`interval` must be [low, high] numbers with low <= high")
            if "render" in m and not _text(m["render"]):
                why.append(at + "`render` must be a non-empty string")
            if "chunk" in m and (not _uint(m["chunk"]) or "render" not in m):
                why.append(at + "`chunk` must be a non-negative integer, with its `render`")

    artifacts = env.get("artifacts")
    if not isinstance(artifacts, list):
        why.append("`artifacts` must be a list")
    else:
        for i, a in enumerate(artifacts):
            at = "artifacts[%d]: " % i
            if not isinstance(a, dict):
                why.append(at + "must be an object")
                continue
            _fields(a, ("path", "type", "sha256"), at, why)
            if not (_text(a.get("path")) and _text(a.get("type")) and _text(a.get("sha256"))
                    and _SHA.match(a["sha256"])):
                why.append(at + "needs a `path`, a media `type` and a lower-case hex `sha256`")
    return why


class Envelope:
    """An envelope under construction. Add metrics and files, then `write()`."""

    def __init__(self, name, version, run, instrument=None, hooks=(), config=None, audio="n/a"):
        self.doc = {
            "format": FORMAT,
            "name": name,
            "version": version,
            "identity": {"run": run, "instrument": instrument or name},
            "provenance": {"config": dict(config or {}), "ggtts": None, "engine": None, "backend": None,
                           "model_sha256": None, "seeds": [], "inputs": [], "audio": audio},
            "hooks": list(hooks),
            "status": "ok",
            "metrics": [],
            "artifacts": [],
            "log": None,
        }

    def from_trace(self, trace, render):
        """Fill the provenance a trace knows: versions, engine, model, and every draw's seed under `render`."""
        h, p = trace.header, self.doc["provenance"]
        p["ggtts"] = h.get("ggtts") or None
        p["engine"] = h.get("engine") or None
        p["backend"] = h.get("backend") or None
        p["model_sha256"] = h.get("model_sha256")
        for e in trace.of("chunk_begin"):
            p["seeds"].append({"render": render, "chunk": e["chunk"], "seed": e["seed"]})
        return self

    def input(self, path):
        """Record a file the instrument read, by hash."""
        self.doc["provenance"]["inputs"].append({"path": os.path.basename(path), "sha256": sha256_of(path)})
        return self

    def metric(self, name, value, unit, direction, interval=None, render=None, chunk=None):
        m = {"name": name, "value": value, "unit": unit, "direction": direction}
        if interval is not None:
            m["interval"] = list(interval)
        if render is not None:
            m["render"] = render
        if chunk is not None:
            m["chunk"] = chunk
        self.doc["metrics"].append(m)
        return self

    def artifact(self, path, media_type):
        self.doc["artifacts"].append({"path": os.path.basename(path), "type": media_type, "sha256": sha256_of(path)})
        return self

    def status(self, status, error=None):
        self.doc["status"] = status
        self.doc["error"] = error
        return self

    def write(self, path):
        """Write the envelope as JSON. Raises `EnvelopeError` instead of writing an invalid one."""
        why = check(self.doc)
        if why:
            raise EnvelopeError(why)
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.doc, f, indent=1, sort_keys=False)
            f.write("\n")
        os.replace(tmp, path)
        return self.doc
