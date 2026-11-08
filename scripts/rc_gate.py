#!/usr/bin/env python3
"""Run CPU checks and both real-engine CUDA probes for promotion."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


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


def evaluate(cpu, gpu, instrument):
    states = list(cpu.values()) + [v["status"] for v in gpu.values()] + [v["status"] for v in instrument.values()]
    if "fail" in states:
        return "fail"
    if (set(cpu) != {"fmt", "clippy", "build", "test"} or set(gpu) != {"qwen", "kitten"}
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="commit/ref to check in a detached temporary worktree")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", default="rc-gate.json")
    parser.add_argument("--work", required=True, help="persistent scratch directory for probe outputs and logs")
    parser.add_argument("--wait-file", help="wait while this GPU-busy marker exists")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    out, scratch = Path(args.out).resolve(), Path(args.work).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    report = {"started": now(), "commit": None, "cpu": {}, "gpu": {}, "instrument": {}, "result": "incomplete",
              "limits": ["GPU probes compare each engine with its own references; this is not a CPU/GPU identity claim.",
                         "Instrument checks compare within each backend; timing and external artifacts are not covered.",
                         "External CUDA library provenance is supplied by the operator; its build cannot be verified here."]}
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
        init = command(["git", "submodule", "update", "--init", "crates/ggtts-ggml-sys/ggml"],
                       work, scratch / "submodule.log")
        if init:
            raise ValueError("ggml submodule initialization failed; see submodule.log")
        for name, cmd in {
            "fmt": ["cargo", "fmt", "--all", "--check"],
            "clippy": ["cargo", "clippy", "--workspace", "--all-targets", "--locked", "-j", str(args.jobs), "--", "-D", "warnings"],
            "build": ["cargo", "build", "--workspace", "--locked", "-j", str(args.jobs)],
            "test": ["cargo", "test", "--workspace", "--locked", "-j", str(args.jobs)],
        }.items():
            report["cpu"][name] = "pass" if command(cmd, work, scratch / (name + ".log")) == 0 else "fail"
        if args.wait_file:
            while Path(args.wait_file).exists():
                print(f"waiting: {args.wait_file}", flush=True)
                time.sleep(30)
        inputs = {
            "qwen": ["QWEN3_TTS_CPP", "QWEN3_TTS_BUILD", "TALKER", "CODEC"],
            "kitten": ["GGML_LIB_DIR", "KITTEN_REPO", "S3_GGUF", "REF_LM", "REF_S3", "REF_COND", "VOICE"],
        }
        scripts = {"qwen": "crates/ggtts-qwen3-tts/scripts/cuda_probe.sh",
                   "kitten": "crates/ggtts-kitten/scripts/cuda_probe.sh"}
        device_log = scratch / "device.log"
        device_ok = command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                            work, device_log) == 0
        report["device_log"] = str(device_log)
        for name, variables in inputs.items():
            entry = {"status": "incomplete", "backend": "cuda", "inputs": {}, "log": str(scratch / (name + ".log"))}
            report["gpu"][name] = entry
            missing = [v for v in variables if not os.environ.get(v) or not Path(os.environ[v]).exists()]
            if not device_ok:
                missing.append("working nvidia-smi / NVIDIA GPU")
            if missing:
                entry["reason"] = "missing: " + ", ".join(missing)
                continue
            try:
                for var in variables:
                    # Build and source trees can contain large generated files; record their location.
                    # Their binary provenance remains an explicit limitation in the report.
                    value = Path(os.environ[var]).resolve()
                    if value == scratch or scratch in value.parents or value in scratch.parents:
                        raise ValueError(f"{var} overlaps scratch; probes may overwrite inputs")
                    entry["inputs"][var] = {"path": str(value)}
                    if var not in ("QWEN3_TTS_CPP", "QWEN3_TTS_BUILD"):
                        entry["inputs"][var]["sha256_tree"] = digest(value)
                if name == "qwen":
                    lib = Path(os.environ["QWEN3_TTS_BUILD"]) / "ggml/src"
                    entry["inputs"]["qwen_ggml_libraries_sha256_tree"] = digest(lib)
                    entry["inputs"]["reference_commit"] = subprocess.check_output(
                        ["git", "-C", os.environ["QWEN3_TTS_CPP"], "rev-parse", "HEAD"], text=True).strip()
                env = os.environ.copy()
                env.update({"WORK": str(scratch / name), "DEVICE": "cuda", "STEPS": "all",
                            "CARGO_TARGET_DIR": str(scratch / name / "target"), "THREADS": str(args.jobs)})
                env.pop("GPU_WAIT_FILE", None)
                cmd = ["bash", scripts[name]]
                entry["command"] = cmd
                entry["status"] = "pass" if command(cmd, work, Path(entry["log"]), env) == 0 else "fail"
                fixtures = scratch / name / "fixtures-cuda"
                if name == "qwen" and fixtures.is_dir():
                    entry["recorded_fixtures_sha256_tree"] = digest(fixtures)
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                entry["reason"] = str(exc)
        # Avoid another expensive model build/run if the preceding promotion checks already failed or are incomplete.
        if (all(v == "pass" for v in report["cpu"].values()) and len(report["cpu"]) == 4
                and len(report["gpu"]) == 2 and all(v["status"] == "pass" for v in report["gpu"].values())):
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
