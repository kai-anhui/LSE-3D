from pathlib import Path
import subprocess
import random
import numpy as np
import torch


def load_checkpoint(path, device="cpu"):
    """Load a trusted local training checkpoint, including RNG/optimizer state."""
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(path, model, optimizer, step, config, threshold=None, sampler=None):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    payload = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict() if optimizer else None,
        step=step,
        config=config,
        threshold=threshold,
        git_hash=commit,
        torch_rng=torch.get_rng_state(),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        sampler_rng=sampler.getstate() if sampler else None,
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
