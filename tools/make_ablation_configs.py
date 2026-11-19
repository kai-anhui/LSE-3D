"""Generate only defined/implemented ablations; no silent baseline substitutions."""

import copy
import yaml
from pathlib import Path
from utils.config import load_config


def main():
    base = load_config("configs/base.yaml")
    variants = {
        "dual_no_gate": {"model": {"gate_mode": "none"}, "loss": {"gate_w": 0}},
        "single_no_pair": {
            "model": {"states": "single", "gate_mode": "none"},
            "loss": {"eq_w": 0, "chg_w": 0, "inc_w": 0, "id_w": 0, "gate_w": 0},
        },
        "single_pair": {
            "model": {"states": "single", "gate_mode": "none"},
            "loss": {"id_w": 0, "gate_w": 0},
        },
        "scope_only": {"model": {"states": "scope_only"}},
        "geometry_gate": {"model": {"gate_use_language": False}},
        "no_pair": {"loss": {"eq_w": 0, "chg_w": 0, "inc_w": 0}},
        "no_gate_aux": {"loss": {"gate_w": 0}},
        "one_scan": {"model": {"directions": 1}},
        "two_scans": {"model": {"directions": 2}},
    }
    folder = Path("configs/ablations")
    folder.mkdir(exist_ok=True)
    for name, changes in variants.items():
        config = copy.deepcopy(base)
        for section, values in changes.items():
            config[section].update(values)
        (folder / (name + ".yaml")).write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
