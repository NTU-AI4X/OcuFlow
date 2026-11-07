from __future__ import annotations

"""Atomic training checkpoints with exact stochastic-state restoration."""

import random
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


STRICT_RESUME_VERSION = 2


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(state)
    if missing:
        raise KeyError(f"checkpoint RNG state is incomplete: missing {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count changed across strict resume: "
                f"checkpoint={len(cuda_state)} current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def save_training_checkpoint(
    path,
    *,
    architecture: str,
    phase: str,
    model,
    ema,
    optimizer,
    step: int,
    epoch: int,
    args,
    component_config: Mapping[str, object],
    data_state: Mapping[str, object],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format": "stage7_three_stage_strict_resume",
        "strict_resume_version": STRICT_RESUME_VERSION,
        "architecture": architecture,
        "step": int(step),
        "epoch": int(epoch),
        "phase": str(phase),
        "args": vars(args),
        "component_config": dict(component_config),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "data_state": dict(data_state),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_checkpoint(
    path,
    *,
    model,
    ema,
    optimizer,
    device,
    component_config: Mapping[str, object],
) -> tuple[int, dict | None, bool]:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved = payload.get("component_config")
    current = dict(component_config)
    if isinstance(saved, Mapping):
        saved = dict(saved)
        # The pre-strict unified trainer hard-coded 2026 and therefore did not
        # serialize a CLI seed in component_config.
        saved.setdefault("seed", 2026)
    if saved != current:
        raise ValueError(f"resume component mismatch: checkpoint={saved} current={current}")
    model.load_state_dict(payload["model"], strict=True)
    ema.load_state_dict(payload["ema"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])

    strict = int(payload.get("strict_resume_version", 0)) == STRICT_RESUME_VERSION
    rng_state = payload.get("rng_state")
    data_state = payload.get("data_state")
    if strict and isinstance(rng_state, Mapping) and isinstance(data_state, Mapping):
        restore_rng_state(rng_state)
        return int(payload.get("step", 0)), dict(data_state), True

    warnings.warn(
        "This checkpoint predates strict-resume state. Model/EMA/optimizer will resume, "
        "but the stochastic training trajectory cannot exactly match an uninterrupted run.",
        RuntimeWarning,
        stacklevel=2,
    )
    return int(payload.get("step", 0)), None, False
