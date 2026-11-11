#!/usr/bin/env python3
"""Run CPU checks and both real-engine CUDA probes for promotion.

Two stages may be taken from earlier evidence instead of rerun (CONTRIBUTING.md, "Promotion"):

* The CPU stage (fmt, clippy, build, test) is taken from CI when every required CI check run on the gated commit
  concluded success. `--cpu-stage` runs it here regardless.
* An engine's CUDA probe is reused from the last passing report when nothing it reads changed since the commit it ran
  on: its crates and their path dependencies, the ggml submodule pin, Cargo.lock, the workspace manifest, the
  toolchain pin, its probe script, and its model files (by sha256). `--no-reuse` reruns every probe.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

# The report's format. A report with any other value is never reused: what it recorded may mean something else.
SCHEMA = "ggtts-rc-gate/2"

# Read by every probe: the workspace manifest and lockfile, the toolchain pin, cargo's own configuration (.cargo/
# can set rustflags, linkers and target directories for every build), ggml (its submodule pointer is a path under
# crates/ggtts-ggml-sys, so a moved pin is a change there), the core crate, and the two top-level directories crates
# read files from (schemas/ by include_str!, fixtures/ by tests).
COMMON_INPUTS = ("Cargo.lock", "Cargo.toml", "rust-toolchain.toml", ".cargo", "crates/ggtts-ggml-sys",
                 "crates/ggtts-core", "schemas", "fixtures")

# One entry per CUDA probe. What a probe's reuse compares (reuse_probe, probe_inputs, machine_inputs):
#   * repository paths: COMMON_INPUTS, `script`, the crates in `packages` with every path dependency they have (from
#     `cargo metadata`, so a new dependency is picked up), and `extra`, the paths outside those crates that the
#     built code reads through CARGO_MANIFEST_DIR;
#   * `env`: the model, reference and library files the probe reads, hashed (paths in ENV_PATHS_ONLY are recorded by
#     commit and library hash instead);
#   * the machine: GPU, driver, the resolved CUDA_HOME and its toolkit version, and RUSTFLAGS/CARGO_ENCODED_RUSTFLAGS.
# The list covers what the probe's build and commands read. It leaves out files only the crates' own tests read and
# the probe does not run, such as scripts/ (read by ggtts-instruments' tests); `extra` errs on the side of including.
# An FFI probe (crates/ggtts-ffi) is meant to become one more entry here, once that crate has a CUDA probe script:
#   "ffi": {"script": "crates/ggtts-ffi/scripts/cuda_probe.sh", "packages": ("ggtts-ffi",), "extra": (), "env": [...]},
# It is not registered: an entry here is required for a pass, so it lands with the script it names.
PROBES = {
    "qwen": {"script": "crates/ggtts-qwen3-tts/scripts/cuda_probe.sh", "packages": ("ggtts-qwen3-tts",),
             "extra": (), "env": ["QWEN3_TTS_CPP", "QWEN3_TTS_BUILD", "TALKER", "CODEC"]},
    # ggtts-cli and ggtts-instruments read the instrument catalogue, the Python SDK and the instrument schema from
    # instruments/ and docs/instruments/.
    "kitten": {"script": "crates/ggtts-kitten/scripts/cuda_probe.sh", "packages": ("ggtts-kitten", "ggtts-cli"),
               "extra": ("instruments", "docs/instruments"),
               "env": ["GGML_LIB_DIR", "KITTEN_REPO", "S3_GGUF", "REF_LM", "REF_S3", "REF_COND", "VOICE"]},
}
# The machine fields every probe's inputs carry. A difference in any of them refuses reuse, naming it.
MACHINE_KEYS = ("gpu", "driver", "cuda_home", "cuda_toolkit", "rustflags", "cargo_encoded_rustflags")
ENV_PATHS_ONLY = ("QWEN3_TTS_CPP", "QWEN3_TTS_BUILD")  # recorded by commit and library hash instead (below)

# CI jobs that run what the gate's CPU stage runs (ci.yml): fmt and clippy, then build and test on both CPU hosts.
# None is selected by scripts/select_specialized_ci.py, so a skipped one is never "not required": it is not a pass.
REQUIRED_CI_CHECKS = ("check (fmt, clippy)", "build-test (ubuntu-latest, CPU)", "build-test (windows-latest, CPU)")
CI_APP = "github-actions"

# Python packages the instrument proof needs, by requirements file: a missing one turns a required instrument into
# "skipped" an hour into the run, so the gate checks them first, with the interpreter the instruments use.
PYTHON_REQUIREMENTS = ("instruments/examples/python/loudness-ends/requirements.txt",)

# The Kitten TQ1_0 model's expected sha256 (plain `sha256sum` of the file). No public TQ1_0 model is published yet, so
# none is pinned here; until one is, the operator supplies it as TQ1_GGUF_SHA256 and the report records it.
TQ1_SHA256 = None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(args, cwd, log, env=None):
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(map(str, args)) + "\n")
        stream.flush()
        try:
            return subprocess.run(args, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        except OSError as exc:
            stream.write(str(exc) + "\n")
            return 127


def digest(path):
    """Hash file contents and relative names, including every file in fixture directories."""
    path = Path(path).resolve(strict=True)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"empty input: {path}")
    h = hashlib.sha256()
    for f in files:
        name = f.name if path.is_file() else f.relative_to(path).as_posix()
        h.update(name.encode() + b"\0")
        with f.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()


def file_sha256(path):
    """The plain sha256 of one file, as `sha256sum` prints it."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


TQ1_NOT_RUN = "TQ1_0 NOT RUN"


def tq1_entry(path, returncode=None, log_text="", expected=None):
    """The Kitten probe's TQ1_0 CUDA checks: pass or fail only when they ran, otherwise incomplete with a reason.

    `path` is TQ1_GGUF when it names an existing file, `returncode` the probe's exit status (None: it did not run),
    `log_text` the probe's log, whose result line says when the TQ1 checks were skipped, and `expected` the model's
    expected sha256. The report carries the model's sha256 beside it: a mismatch fails, and checks that passed on a
    model with no expected value are incomplete, since nothing says which model they passed on.
    """
    if path is None:
        return {"status": "incomplete", "reason": "tq1: not run (TQ1_GGUF is not set or is not a file)"}
    entry = {"input": {"path": str(path), "sha256_tree": digest(path)}, "sha256": file_sha256(path),
             "expected_sha256": expected}
    if expected and entry["sha256"] != expected.lower():
        entry.update(status="fail", reason=f"tq1: model sha256 {entry['sha256']} is not the expected {expected}")
    elif returncode is None:
        entry.update(status="incomplete", reason="tq1: not run (the Kitten probe did not run)")
    elif returncode != 0:
        entry["status"] = "fail"
    elif TQ1_NOT_RUN in log_text:
        entry.update(status="incomplete", reason="tq1: not run (the probe's result line says so)")
    elif not expected:
        entry.update(status="incomplete",
                     reason="tq1: the checks passed, but no expected sha256 was given (set TQ1_GGUF_SHA256)")
    else:
        entry["status"] = "pass"
    return entry


def evaluate(cpu, gpu, instrument):
    states = list(cpu.values()) + [v["status"] for v in gpu.values()] + [v["status"] for v in instrument.values()]
    # The Kitten probe's TQ1_0 checks are reported beside it: unrun, they keep the gate from a pass.
    states.append(gpu.get("kitten", {}).get("tq1", {}).get("status", "incomplete"))
    if "fail" in states:
        return "fail"
    if (set(cpu) != {"fmt", "clippy", "build", "test"} or set(gpu) != set(PROBES)
            or set(instrument) != {"cpu", "cuda"}):
        return "incomplete"
    return "pass" if states and all(s == "pass" for s in states) else "incomplete"


def instrument_proof(work, scratch, device_ok, jobs):
    # Load the proof from the checked ref, not from the checkout that launched --ref.
    spec = importlib.util.spec_from_file_location("checked_instrument_gate", work / "scripts/instrument_promotion_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = os.environ.get("QWEN3_TTS_BUILD")
    cuda_lib = Path(build) / "ggml/src" if build else None
    return module.run(work, scratch, os.environ.get("TALKER"), os.environ.get("CODEC"), cuda_lib,
                      device_ok, jobs, command)


# ---- Python requirements -------------------------------------------------------------------------------------------

_VERSIONS = """import json, sys
from importlib import metadata
found = {}
for name in json.loads(sys.argv[1]):
    try:
        found[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        found[name] = None
print(json.dumps(found))
"""


def read_requirements(path):
    """`name==version` (or a bare `name`) per line of a requirements file; comments and blank lines ignored."""
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*(?:==\s*([A-Za-z0-9_.+-]+))?", line)
        if not match:
            raise ValueError(f"{path}: cannot read requirement {line!r}")
        pins[match.group(1)] = match.group(2)
    return pins


def instrument_python():
    """The interpreter the instruments run under (crates/ggtts-instruments/src/runner.rs): GGTTS_PYTHON, else python3."""
    return os.environ.get("GGTTS_PYTHON") or shutil.which("python3") or "python3"


def python_requirements(work, interpreter):
    """Each pinned package for `interpreter`: found or not, and at which version. `problems` names what is wrong."""
    wanted = {}
    for rel in PYTHON_REQUIREMENTS:
        for name, version in read_requirements(Path(work) / rel).items():
            wanted[name] = (version, rel)
    result = {"interpreter": interpreter, "packages": {}, "problems": []}
    try:
        proc = subprocess.run([interpreter, "-c", _VERSIONS, json.dumps(sorted(wanted))],
                              capture_output=True, text=True, timeout=120)
        found = json.loads(proc.stdout) if proc.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        found, proc = None, None
        result["problems"].append(f"cannot run {interpreter}: {exc}")
    if found is None:
        if proc is not None:
            result["problems"].append(f"cannot read installed packages with {interpreter}: {proc.stderr.strip()}")
        return result
    for name, (version, rel) in sorted(wanted.items()):
        got = found.get(name)
        result["packages"][name] = {"required": version, "found": got, "requirements": rel}
        if got is None:
            result["problems"].append(f"{name} is not installed for {interpreter} (required by {rel}; "
                                      f"install with: {interpreter} -m pip install -r {rel})")
        elif version and got != version:
            result["problems"].append(f"{name} {got} is installed for {interpreter}; {rel} requires {version}")
    return result


# ---- CPU stage from CI ---------------------------------------------------------------------------------------------

def gh_api(path):
    return json.loads(subprocess.check_output(["gh", "api", "-H", "Accept: application/vnd.github+json", path],
                                              text=True))


def github_repo(root):
    url = subprocess.check_output(["git", "-C", str(root), "remote", "get-url", "origin"], text=True).strip()
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", url)
    if not match:
        raise ValueError(f"origin is not a GitHub repository: {url}")
    return match.group(1)


def ci_cpu_checks(repo, sha, api=gh_api):
    """Whether CI already ran the CPU stage on `sha`: (True, detail) only when every required check run succeeded.

    Each required name must have at least one GitHub Actions run on the commit, every latest run of that name must be
    completed with conclusion success. Skipped, neutral, cancelled, pending or absent is not a pass.
    """
    detail = {"commit": sha, "repo": repo, "checks": {}}
    ok = True
    for name in REQUIRED_CI_CHECKS:
        try:
            data = api(f"repos/{repo}/commits/{sha}/check-runs?check_name={_quote(name)}&filter=latest&per_page=100")
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            detail["checks"][name] = f"unreadable ({exc})"
            ok = False
            continue
        runs = [r for r in data.get("check_runs", [])
                if r.get("name") == name and (r.get("app") or {}).get("slug") == CI_APP]
        if not runs:
            detail["checks"][name] = "absent"
            ok = False
            continue
        states = sorted({r.get("conclusion") if r.get("status") == "completed" else r.get("status") for r in runs},
                        key=str)
        detail["checks"][name] = ",".join(map(str, states))
        if states != ["success"]:
            ok = False
    return ok, detail


def _quote(text):
    from urllib.parse import quote
    return quote(text, safe="")


# ---- Probe reuse ---------------------------------------------------------------------------------------------------

def crate_paths(work, name):
    """The directories of the crates probe `name` builds and of every path dependency they have (cargo metadata)."""
    meta = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"],
                                              cwd=work, text=True))
    work = Path(work).resolve()
    dirs = {p["name"]: Path(p["manifest_path"]).resolve().parent for p in meta["packages"]}
    deps = {p["name"]: [d["name"] for d in p["dependencies"] if d.get("path")] for p in meta["packages"]}
    seen, todo = set(), list(PROBES[name]["packages"])
    while todo:
        package = todo.pop()
        if package in seen:
            continue
        if package not in dirs:
            raise ValueError(f"cargo metadata has no package {package}")
        seen.add(package)
        todo.extend(deps.get(package, []))
    return {dirs[p].relative_to(work).as_posix() for p in seen}


def probe_paths(work, name):
    """Repository paths a probe reads: COMMON_INPUTS, its script, its `extra` paths, and its crates (crate_paths)."""
    probe = PROBES[name]
    return sorted(set(COMMON_INPUTS) | {probe["script"]} | set(probe.get("extra", ())) | crate_paths(work, name))


def fingerprint(inputs):
    """What an input record says about content, without the paths it was read from."""
    out = {}
    for key, value in inputs.items():
        if isinstance(value, dict):
            if "sha256_tree" in value:
                out[key] = value["sha256_tree"]
        else:
            out[key] = value
    return out


def load_previous(path):
    """The earlier report, or (None, why it cannot be reused)."""
    if path is None:
        return None, "no earlier report"
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"no earlier report at {path}"
    except (OSError, ValueError) as exc:
        return None, f"earlier report {path} is unreadable: {exc}"
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        got = report.get("schema") if isinstance(report, dict) else None
        return None, f"earlier report {path} has schema {got!r}, not {SCHEMA!r}"
    if report.get("result") != "pass":
        return None, f"earlier report {path} did not pass (result {report.get('result')!r})"
    if not report.get("commit"):
        return None, f"earlier report {path} names no commit"
    return report, None


def reuse_probe(root, previous, previous_path, name, commit, paths, inputs, tq1=None):
    """The earlier passing entry for probe `name`, marked reused, or (None, why not).

    Reused only when the earlier report passed with this schema, its entry for the probe passed, none of `paths`
    differs between the commit the probe actually ran on and `commit`, and every hashed model input (and the TQ1
    model, for Kitten) has the same sha256 as then.
    """
    if previous is None:
        return None, "no passing earlier report"
    old = previous.get("gpu", {}).get(name)
    if not isinstance(old, dict) or old.get("status") != "pass":
        return None, f"the earlier report's {name} probe did not pass"
    origin = old.get("reused") or {"commit": previous["commit"], "report": str(previous_path)}
    base = origin.get("commit")
    # A machine field that could not be read proves nothing by matching: two runs without nvcc, or without a readable
    # nvidia-smi answer, compare equal while the card, or the libraries behind CUDA_HOME, may have changed.
    unread = []
    for key in MACHINE_KEYS:
        sides = [label for label, side in (("earlier", old.get("inputs", {})), ("now", inputs))
                 if str(side.get(key, "")).startswith("unavailable")]
        if sides:
            unread.append(f"{key} ({' and '.join(sides)})")
    if unread:
        return None, "the machine cannot be compared, a field could not be read: " + ", ".join(unread)
    before, after = fingerprint(old.get("inputs", {})), fingerprint(inputs)
    differ = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if differ:
        machine = [k for k in differ if k in MACHINE_KEYS]
        files = [k for k in differ if k not in MACHINE_KEYS]
        parts = []
        if machine:
            parts.append("the machine differs from the earlier report's: " + ", ".join(
                f"{k} {before.get(k)!r} -> {after.get(k)!r}" for k in machine))
        if files:
            parts.append("a model or reference input differs from the earlier report's (sha256): " + ", ".join(files))
        return None, "; ".join(parts)
    if name == "kitten":
        old_tq1 = old.get("tq1") or {}
        if (old_tq1.get("status") != "pass" or tq1 is None or not tq1.get("expected_sha256")
                or tq1.get("sha256") != old_tq1.get("sha256")):
            return None, ("the TQ1_0 model differs from the earlier report's, has no expected sha256, or its checks "
                          "did not pass there")
    try:
        subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{base}^{{commit}}"], check=True,
                       capture_output=True)
        changed = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", base, commit, "--", *paths],
                                          text=True).split()
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, f"cannot compare {base} with {commit}: {exc}"
    if changed:
        shown = ", ".join(changed[:5]) + (f" and {len(changed) - 5} more" if len(changed) > 5 else "")
        return None, f"inputs changed since {base}: {shown}"
    entry = copy.deepcopy(old)
    entry["inputs"] = inputs
    entry["reused"] = {"commit": base, "report": origin.get("report"), "paths": paths}
    entry["note"] = f"reused from {base} ({origin.get('report')})"
    if tq1 is not None:
        entry["tq1"] = dict(old.get("tq1"), **{"input": tq1["input"], "expected_sha256": tq1.get("expected_sha256")})
    return entry, None


def device_identity(log):
    """GPU names and driver versions from the gate's `nvidia-smi --query-gpu=name,driver_version` log."""
    try:
        lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None, None
    rows = [line.rsplit(",", 1) for line in lines if "," in line and not line.startswith("$ ")]
    if not rows:
        return None, None
    return " | ".join(r[0].strip() for r in rows), " | ".join(r[1].strip() for r in rows)


def cuda_toolkit(cuda_home):
    """The toolkit's own version lines from `<CUDA_HOME>/bin/nvcc --version`."""
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    try:
        proc = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({type(exc).__name__})"
    if proc.returncode:
        return f"unavailable (nvcc exited {proc.returncode})"
    lines = [line.strip() for line in proc.stdout.splitlines()
             if line.strip().startswith(("Cuda compilation tools", "Build "))]
    return " / ".join(lines) or proc.stdout.strip()


def machine_inputs(device_ok, device_log):
    """What the probes ran on and were built with, besides the repository and the model files.

    The probes put CUDA_HOME/bin on PATH and link the CUDA runtime from it (default /usr/local/cuda, resolved, so a
    re-pointed symlink is a different toolkit), and cargo applies RUSTFLAGS / CARGO_ENCODED_RUSTFLAGS to every build.
    """
    gpu, driver = device_identity(device_log) if device_ok else (None, None)
    cuda_home = os.path.realpath(os.environ.get("CUDA_HOME") or "/usr/local/cuda")
    return {"gpu": gpu or "unavailable", "driver": driver or "unavailable", "cuda_home": cuda_home,
            "cuda_toolkit": cuda_toolkit(cuda_home), "rustflags": os.environ.get("RUSTFLAGS", ""),
            "cargo_encoded_rustflags": os.environ.get("CARGO_ENCODED_RUSTFLAGS", "")}


def probe_inputs(name, scratch):
    """Hash what probe `name` reads from the environment, refusing anything that overlaps the scratch directory."""
    inputs = {}
    for var in PROBES[name]["env"]:
        # Build and source trees can contain large generated files; record their location.
        # Their binary provenance remains an explicit limitation in the report.
        value = Path(os.environ[var]).resolve()
        if value == scratch or scratch in value.parents or value in scratch.parents:
            raise ValueError(f"{var} overlaps scratch; probes may overwrite inputs")
        inputs[var] = {"path": str(value)}
        if var not in ENV_PATHS_ONLY:
            inputs[var]["sha256_tree"] = digest(value)
    if name == "qwen":
        lib = Path(os.environ["QWEN3_TTS_BUILD"]) / "ggml/src"
        inputs["qwen_ggml_libraries_sha256_tree"] = digest(lib)
        inputs["reference_commit"] = subprocess.check_output(
            ["git", "-C", os.environ["QWEN3_TTS_CPP"], "rev-parse", "HEAD"], text=True).strip()
    return inputs


# ---- The gate ------------------------------------------------------------------------------------------------------

CPU_COMMANDS = ("fmt", "clippy", "build", "test")


def cpu_stage(args, work, scratch, report, root):
    """Take the CPU stage from CI when it is green on the commit, else run it here."""
    if not args.cpu_stage:
        try:
            repo = args.repo or github_repo(root)
            ok, detail = ci_cpu_checks(repo, report["commit"])
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            ok, detail = False, {"error": str(exc)}
        if ok:
            report["cpu"] = dict.fromkeys(CPU_COMMANDS, "pass")
            report["cpu_source"] = dict(detail, mode="ci")
            return
        report["cpu_source"] = dict(detail, mode="local", reason="CI is not green on this commit for every required "
                                                                 "check, so the CPU stage runs here")
    else:
        report["cpu_source"] = {"mode": "local", "reason": "--cpu-stage"}
    for name, cmd in {
        "fmt": ["cargo", "fmt", "--all", "--check"],
        "clippy": ["cargo", "clippy", "--workspace", "--all-targets", "--locked", "-j", str(args.jobs), "--", "-D", "warnings"],
        "build": ["cargo", "build", "--workspace", "--locked", "-j", str(args.jobs)],
        "test": ["cargo", "test", "--workspace", "--locked", "-j", str(args.jobs)],
    }.items():
        report["cpu"][name] = "pass" if command(cmd, work, scratch / (name + ".log")) == 0 else "fail"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", help="commit/ref to check in a detached temporary worktree")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", default="rc-gate.json")
    parser.add_argument("--work", required=True, help="persistent scratch directory for probe outputs and logs")
    parser.add_argument("--wait-file", help="wait while this GPU-busy marker exists")
    parser.add_argument("--previous", help="the last passing report, whose probes may be reused (default: --out, "
                                           "read before it is overwritten)")
    parser.add_argument("--no-reuse", action="store_true", help="rerun every CUDA probe")
    parser.add_argument("--cpu-stage", action="store_true", help="run fmt, clippy, build and test here even when CI "
                                                                 "is green on the commit")
    parser.add_argument("--repo", help="owner/name for the CI check-run lookup (default: from origin)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    out, scratch = Path(args.out).resolve(), Path(args.work).resolve()
    previous_path = Path(args.previous).resolve() if args.previous else out
    if args.no_reuse:
        previous, previous_why = None, "--no-reuse"
    else:
        previous, previous_why = load_previous(previous_path)
        if previous is not None:
            # A copy, so a reused entry still names a report that exists after --out is overwritten.
            kept = scratch / f"previous-{previous['commit'][:12]}.json"
            scratch.mkdir(parents=True, exist_ok=True)
            if previous_path != kept:
                shutil.copyfile(previous_path, kept)
            previous_path = kept
    scratch.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "started": now(), "commit": None, "cpu": {}, "gpu": {}, "instrument": {},
              "result": "incomplete",
              "reuse": {"previous": str(previous_path) if previous else None, "refused": previous_why},
              "limits": ["GPU probes compare each engine with its own references; this is not a CPU/GPU identity claim.",
                         "The Kitten TQ1_0 CUDA checks need TQ1_GGUF and its expected sha256 (TQ1_GGUF_SHA256); without them gpu.kitten.tq1 is incomplete, and so is the result.",
                         "Instrument checks compare within each backend; timing and external artifacts are not covered.",
                         "External CUDA library provenance is supplied by the operator; its build cannot be verified here.",
                         "A reused probe ran on the commit it names; it is reused only on the same GPU, driver, CUDA toolkit and RUSTFLAGS, and inputs outside its listed paths, hashed models and those machine fields are not compared."]}
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    work = root
    tmp = None
    exit_code = 1
    try:
        if args.ref:
            tmp = Path(tempfile.mkdtemp(prefix="ggtts-rc-gate-"))
            work = tmp / "checkout"
            subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", str(work), args.ref], check=True)
        elif subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=work, text=True).strip():
            raise ValueError("checkout has uncommitted or untracked files; use --ref")
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work, text=True).strip()
        report["ref"] = args.ref or "HEAD"
        report["rustc"] = subprocess.check_output(["rustc", "--version"], cwd=work, text=True).strip()
        pinned = next(line.split('"')[1] for line in (work / "rust-toolchain.toml").read_text().splitlines()
                      if line.strip().startswith("channel"))
        if report["rustc"].split()[1] != pinned:
            raise ValueError(f"rustc does not match pinned version {pinned}")
        report["python"] = python_requirements(work, instrument_python())
        if report["python"]["problems"]:
            raise ValueError("python requirements: " + "; ".join(report["python"]["problems"]))
        init = command(["git", "submodule", "update", "--init", "crates/ggtts-ggml-sys/ggml"],
                       work, scratch / "submodule.log")
        if init:
            raise ValueError("ggml submodule initialization failed; see submodule.log")
        cpu_stage(args, work, scratch, report, root)
        if args.wait_file:
            while Path(args.wait_file).exists():
                print(f"waiting: {args.wait_file}", flush=True)
                time.sleep(30)
        device_log = scratch / "device.log"
        device_ok = command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                            work, device_log) == 0
        report["device_log"] = str(device_log)
        report["machine"] = machine = machine_inputs(device_ok, device_log)
        for name, probe in PROBES.items():
            variables = probe["env"]
            entry = {"status": "incomplete", "backend": "cuda", "inputs": {}, "log": str(scratch / (name + ".log"))}
            report["gpu"][name] = entry
            tq1 = None
            expected = os.environ.get("TQ1_GGUF_SHA256") or TQ1_SHA256
            if name == "kitten":
                given = os.environ.get("TQ1_GGUF")
                tq1 = Path(given).resolve() if given and Path(given).is_file() else None
                try:
                    if tq1 and (scratch in tq1.parents or tq1 == scratch):
                        raise ValueError("TQ1_GGUF overlaps scratch; probes may overwrite inputs")
                    entry["tq1"] = tq1_entry(tq1, expected=expected)
                except (OSError, ValueError) as exc:
                    tq1 = None
                    entry["tq1"] = {"status": "incomplete", "reason": f"tq1: not run ({exc})"}
                if entry["tq1"].get("status") == "fail":
                    tq1 = None  # a model that is not the expected one is not worth the probe's time
            missing = [v for v in variables if not os.environ.get(v) or not Path(os.environ[v]).exists()]
            if missing:
                entry["reason"] = "missing: " + ", ".join(missing)
                continue
            try:
                entry["inputs"] = dict(probe_inputs(name, scratch), **machine)
                if previous is not None:
                    try:
                        paths = probe_paths(work, name)
                    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
                        reused, why = None, f"cannot list the probe's inputs: {exc}"
                    else:
                        reused, why = reuse_probe(root, previous, previous_path, name, report["commit"], paths,
                                                  entry["inputs"], entry.get("tq1") if tq1 else None)
                    if reused is not None:
                        report["gpu"][name] = reused
                        continue
                    entry["reuse_refused"] = why
                if not device_ok:
                    entry["reason"] = "missing: working nvidia-smi / NVIDIA GPU"
                    continue
                env = os.environ.copy()
                env.update({"WORK": str(scratch / name), "DEVICE": "cuda", "STEPS": "all",
                            "CARGO_TARGET_DIR": str(scratch / name / "target"), "THREADS": str(args.jobs)})
                env.pop("GPU_WAIT_FILE", None)
                env.pop("TQ1_GGUF", None)
                if tq1:
                    env["TQ1_GGUF"] = str(tq1)
                cmd = ["bash", probe["script"]]
                entry["command"] = cmd
                code = command(cmd, work, Path(entry["log"]), env)
                entry["status"] = "pass" if code == 0 else "fail"
                if tq1:
                    log = Path(entry["log"])
                    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
                    entry["tq1"] = tq1_entry(tq1, code, text, expected)
                fixtures = scratch / name / "fixtures-cuda"
                if name == "qwen" and fixtures.is_dir():
                    entry["recorded_fixtures_sha256_tree"] = digest(fixtures)
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                entry["reason"] = str(exc)
        # Avoid another expensive model build/run if the preceding promotion checks already failed or are incomplete.
        if (all(v == "pass" for v in report["cpu"].values()) and len(report["cpu"]) == len(CPU_COMMANDS)
                and set(report["gpu"]) == set(PROBES) and all(v["status"] == "pass" for v in report["gpu"].values())):
            report["instrument"] = instrument_proof(work, scratch, device_ok, args.jobs)
        report["result"] = evaluate(report["cpu"], report["gpu"], report["instrument"])
        exit_code = 0 if report["result"] == "pass" else 1
    except (OSError, ValueError, StopIteration, subprocess.CalledProcessError) as exc:
        report["error"] = str(exc)
    finally:
        if tmp:
            subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(work)], check=False)
            shutil.rmtree(tmp, ignore_errors=True)
        report["finished"] = now()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
