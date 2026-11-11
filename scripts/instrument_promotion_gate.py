"""Run the pinned real-Qwen instrument identity proofs for a promotion gate.

The CPU and CUDA runs compare within each backend. They do not prove cross-backend parity, timing identity,
or unchanged external artifacts. The model and libraries are inputs, never written here.
"""
import hashlib
import os
from pathlib import Path

TALKER_SHA = "33502b2035f97d7f458248d91e965a6800026549d382c5e8e300176027c70b67"
CODEC_SHA = "7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(checkout, scratch, talker, codec, cuda_lib, device_ok, jobs, command):
    """Return explicit CPU/CUDA proof states; command logs and returns its process exit code."""
    result = {name: {"status": "incomplete", "backend": name, "steps": {}} for name in ("cpu", "cuda")}
    files = {"talker": (talker, TALKER_SHA), "codec": (codec, CODEC_SHA)}
    for name, (path, expected) in files.items():
        if not path or not Path(path).is_file():
            for entry in result.values():
                entry["reason"] = f"missing pinned {name}"
            return result
        got = file_sha(path)
        if got != expected:
            for entry in result.values():
                entry.update(status="fail", reason=f"{name} SHA-256 does not match the published v0.0.1 pin")
            return result
    checkout, scratch = Path(checkout).resolve(), Path(scratch).resolve()
    for path, _ in files.values():
        path = Path(path).resolve()
        if path == scratch or scratch in path.parents or path in scratch.parents:
            raise ValueError("instrument proof scratch overlaps a model input")
    if not device_ok or not cuda_lib or not Path(cuda_lib).is_dir():
        result["cuda"]["reason"] = "missing working GPU or Qwen CUDA libraries"

    for backend in ("cpu", "cuda"):
        entry = result[backend]
        entry["models"] = {"talker_sha256": TALKER_SHA, "codec_sha256": CODEC_SHA}
        if backend == "cuda" and "reason" in entry:
            continue
        env = os.environ.copy()
        env.update({"GGTTS_TALKER": str(Path(talker).resolve()), "GGTTS_CODEC": str(Path(codec).resolve()),
                    "GGTTS_DEVICE": backend, "GGML_SRC": str(checkout / "crates/ggtts-ggml-sys/ggml"),
                    "CARGO_TARGET_DIR": str(scratch / (backend + "-target"))})
        if backend == "cpu":
            env.pop("GGML_LIB_DIR", None)  # build the pinned checkout's ggml, not an arbitrary inherited library
        else:
            env["GGML_LIB_DIR"] = str(Path(cuda_lib).resolve())
        cli_feature = "ggtts-cli/ggml" if backend == "cpu" else "ggtts-cli/cuda"
        qwen_feature = "ggml,plant" if backend == "cpu" else "cuda,plant"
        tests = [
            ("events", "ggtts-cli", cli_feature.replace("ggtts-cli/", ""), "no_change_real",
             "each_event_instrument_preserves_real_qwen_tokens_and_samples"),
            ("stop_trace", "ggtts-qwen3-tts", qwen_feature, "frame_hook",
             "stop_trace_tensor_hook_preserves_real_tokens_and_samples"),
            ("logit_plant", "ggtts-qwen3-tts", qwen_feature, "frame_hook",
             "a_hook_boundary_logit_mutation_is_detected_on_real_qwen"),
        ]
        build = ["cargo", "build", "--locked", "--release", "-j", str(jobs), "-p", "ggtts-cli", "-p",
                 "ggtts-instruments", "--features", cli_feature]
        steps = [("build", build, None)] + [
            (label, ["cargo", "test", "--locked", "--release", "-j", str(jobs), "-p", package, "--features",
                     features, "--test", target, case, "--", "--exact", "--ignored", "--nocapture"], case)
            for label, package, features, target, case in tests
        ]
        for label, cmd, case in steps:
            log = scratch / f"instrument-{backend}-{label}.log"
            rc = command(cmd, checkout, log, env)
            # Cargo can exit zero when a filter matches zero tests. A passing proof must run the named test.
            text = log.read_text(encoding="utf-8") if log.is_file() else ""
            seen = case is None or (case in text and "test result: ok. 1 passed; 0 failed" in text)
            entry["steps"][label] = {"status": "pass" if rc == 0 and seen else "fail", "log": str(log),
                                      "command": cmd}
            if rc or not seen:
                entry.update(status="fail", reason=f"{label} failed or did not execute the named test")
                break
        else:
            entry["status"] = "pass"
        if entry["status"] == "fail":
            if backend == "cpu":
                result["cuda"].setdefault("reason", "not run because the CPU proof failed")
            break  # do not spend GPU time after a CPU failure
    return result
