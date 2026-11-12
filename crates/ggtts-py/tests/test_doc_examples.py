"""The Python examples in docs/bindings/examples/python, each run as a script on the fake engine with the installed
ggtts, so an example the documentation shows cannot stop working unnoticed."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "docs" / "bindings" / "examples" / "python"
SCRIPTS = sorted(EXAMPLES.glob("*.py"))


def test_the_examples_are_found():
    assert {p.name for p in SCRIPTS} >= {"first_render.py", "stream.py"}, EXAMPLES


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_an_example_runs_on_the_fake_engine(script, tmp_path):
    # UTF-8 output, so a message quoting Chinese text prints on a Windows console code page too.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    run = subprocess.run([sys.executable, str(script), "fake"], cwd=tmp_path, capture_output=True, text=True,
                         encoding="utf-8", env=env, timeout=300)
    assert run.returncode == 0, f"{script.name} exited {run.returncode}\n{run.stdout}\n{run.stderr}"
    if script.name == "first_render.py":
        assert (tmp_path / "first_render.flac").stat().st_size > 0
