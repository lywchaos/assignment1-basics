from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap

import psutil
import pytest

from cs336_basics import p9_train_bpe_tinystories as runner


def test_sigint_terminates_child_process() -> None:
    helper = textwrap.dedent(
        """
        import os
        import signal
        import subprocess
        import sys

        from cs336_basics.p9_train_bpe_tinystories import _ProcessTreeSignalHandler

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        print(child.pid, flush=True)
        handler = _ProcessTreeSignalHandler(terminate_timeout_seconds=1.0, kill_timeout_seconds=1.0)

        try:
            with handler:
                os.kill(os.getpid(), signal.SIGINT)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("SIGINT did not interrupt the parent")

        child.wait(timeout=2.0)
        raise SystemExit(128 + signal.SIGINT)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", helper],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )

    assert result.returncode == 128 + signal.SIGINT, result.stderr
    child_pid = int(result.stdout.strip())
    assert not psutil.pid_exists(child_pid)


def test_interrupted_training_is_recorded_in_worklog(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("A tiny training input.<|endoftext|>", encoding="utf-8")
    output_dir = tmp_path / "output"

    def interrupt_training(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_run_train", interrupt_training)

    with pytest.raises(KeyboardInterrupt):
        runner.train_tinystories(
            input_path=input_path,
            output_dir=output_dir,
            vocab_size=257,
            show_progress=False,
        )

    worklog = json.loads((output_dir / runner.WORKLOG_FILENAME).read_text(encoding="utf-8"))
    assert worklog["status"] == "interrupted"
    assert worklog["phase"] == "interrupted"
    assert worklog["error"] == {
        "type": "KeyboardInterrupt",
        "message": "Interrupted by SIGINT",
        "phase": "training",
        "signal": "SIGINT",
    }
