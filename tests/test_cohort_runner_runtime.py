from __future__ import annotations

import importlib.util
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cohort_runtime", ROOT / "scripts/run_subject_cohort.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_child_environment_matches_original_launchers(monkeypatch):
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    for name in names:
        monkeypatch.setenv(name, "12")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    environment = runner.child_environment()
    assert all(environment[name] == "1" for name in names)
    assert environment["PYTHONFAULTHANDLER"] == environment["PYTHONUNBUFFERED"] == "1"
    assert environment["MPLBACKEND"] == "Agg"
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["PYTHONPATH"] == str(ROOT / "src")


def test_log_retains_stdout_stderr_and_refuses_overwrite(tmp_path, capsys):
    path = tmp_path / "child.log"
    command = [sys.executable, "-c", "import sys; print('OUT'); print('ERR', file=sys.stderr)"]
    runner.run_logged(command, runner.child_environment(), path)
    assert "OUT" in path.read_text() and "ERR" in path.read_text()
    assert "OUT" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        runner.run_logged(command, runner.child_environment(), path)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal exit codes")
def test_native_signal_retains_log_and_stops_queue(tmp_path):
    path = tmp_path / "fatal.log"
    command = [sys.executable, "-c", "import os, signal; print('BEFORE_SIGNAL', flush=True); os.kill(os.getpid(), signal.SIGTERM)"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        runner.run_logged(command, runner.child_environment(), path)
    assert error.value.returncode == -signal.SIGTERM
    assert "BEFORE_SIGNAL" in path.read_text()
