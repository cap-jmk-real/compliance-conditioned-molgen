"""Checkpoint and progress helpers for Colab runs.

Written for the execution rules in OBJECTIVES_AND_MODELS.md, Section 5: checkpoints land on
mounted Google Drive, writes are atomic, every script resumes from the latest checkpoint by
default, and progress is printed and flushed so a watching human can see the run is alive.

No training code, no model framework, no network access. The save and load callables are passed
in, so this module stays framework agnostic.

Typical use:

    ck = CheckpointStore("/content/drive/MyDrive/compliance-molgen/runs/uv-chemprop-s0")
    state = ck.latest()            # None on a fresh run
    start_step = state["step"] if state else 0
    hb = Heartbeat(total_steps=10000, every=50)
    ...
    ck.save({"step": step, "loss": loss}, step=step, metric=val_mae, lower_is_better=True)
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import time
from typing import Any, Callable, Optional


def _default_save(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle)


def _default_load(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class CheckpointStore:
    """Atomic checkpoint writes with retention of the last N plus the best."""

    def __init__(
        self,
        run_dir: str,
        keep_last: int = 3,
        suffix: str = ".json",
        save_fn: Callable[[Any, str], None] = _default_save,
        load_fn: Callable[[str], Any] = _default_load,
    ) -> None:
        self.run_dir = pathlib.Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.suffix = suffix
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.meta_path = self.run_dir / "checkpoints.json"

    def _meta(self) -> dict:
        if self.meta_path.exists():
            with open(self.meta_path, encoding="utf-8") as handle:
                return json.load(handle)
        return {"steps": [], "best": None, "best_metric": None}

    def _write_meta(self, meta: dict) -> None:
        tmp = self.meta_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        os.replace(tmp, self.meta_path)

    def path_for(self, step: int) -> pathlib.Path:
        return self.run_dir / f"step-{step:09d}{self.suffix}"

    def save(
        self,
        state: Any,
        step: int,
        metric: Optional[float] = None,
        lower_is_better: bool = True,
    ) -> str:
        """Write a checkpoint atomically and prune old ones. Returns the path written."""
        target = self.path_for(step)
        tmp = target.with_name(target.name + ".tmp")
        self.save_fn(state, str(tmp))
        os.replace(tmp, target)

        meta = self._meta()
        meta["steps"] = sorted(set(meta["steps"] + [step]))
        if metric is not None:
            best = meta.get("best_metric")
            better = (
                best is None
                or (lower_is_better and metric < best)
                or (not lower_is_better and metric > best)
            )
            if better:
                meta["best_metric"] = metric
                meta["best"] = step
                best_path = self.run_dir / f"best{self.suffix}"
                tmp_best = best_path.with_name(best_path.name + ".tmp")
                shutil.copyfile(target, tmp_best)
                os.replace(tmp_best, best_path)
        self._write_meta(meta)
        self._prune(meta)
        return str(target)

    def _prune(self, meta: dict) -> None:
        keep = set(meta["steps"][-self.keep_last :])
        if meta.get("best") is not None:
            keep.add(meta["best"])
        for step in list(meta["steps"]):
            if step not in keep:
                path = self.path_for(step)
                if path.exists():
                    path.unlink()
                meta["steps"].remove(step)
        self._write_meta(meta)

    def latest_path(self) -> Optional[str]:
        meta = self._meta()
        for step in reversed(meta["steps"]):
            path = self.path_for(step)
            if path.exists():
                return str(path)
        return None

    def latest(self) -> Optional[Any]:
        """Load the newest checkpoint, or None on a fresh run. Resume is the default."""
        path = self.latest_path()
        if path is None:
            return None
        return self.load_fn(path)


class Heartbeat:
    """Prints a flushed progress line every `every` steps."""

    def __init__(self, total_steps: Optional[int] = None, every: int = 50, stream=sys.stdout):
        self.total_steps = total_steps
        self.every = every
        self.stream = stream
        self.start = time.time()
        self.last_step = 0

    def tick(self, step: int, **fields: Any) -> None:
        if step % self.every and step != self.total_steps:
            return
        elapsed = time.time() - self.start
        done = max(step - self.last_step, 1)
        rate = step / elapsed if elapsed > 0 else 0.0
        parts = [f"step {step}"]
        if self.total_steps:
            parts.append(f"of {self.total_steps}")
            if rate > 0:
                remaining = (self.total_steps - step) / rate
                parts.append(f"eta {remaining / 60:.1f} min")
        parts.append(f"elapsed {elapsed / 60:.1f} min")
        parts.append(f"{rate:.2f} steps/s")
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        print(" | ".join(parts), file=self.stream, flush=True)
        self.last_step = step
