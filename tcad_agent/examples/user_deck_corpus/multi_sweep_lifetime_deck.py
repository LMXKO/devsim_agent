from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("ACTSOFT_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcad_agent.examples.user_deck_corpus.common import run_public_pn_deck


geometry = {
    "length_um": 0.1,
    "junction_um": 0.05,
}

doping = {
    "p_doping_cm3": 8.0e17,
    "n_doping_cm3": 9.0e17,
}

physics_models = {
    "electron_lifetime_s": 5.0e-9,
    "hole_lifetime_s": 8.0e-9,
    "temperature_k": 300.0,
}

mesh = {
    "contact_spacing_um": 0.001,
    "junction_spacing_um": 1.0e-5,
}

bias = {
    "sweeps": [
        {"name": "low_forward", "start": 0.0, "stop": 0.1, "step": 0.1},
        {"name": "nominal_forward", "start": 0.1, "stop": 0.2, "step": 0.1},
    ],
}


def deck_config() -> dict[str, object]:
    return {
        "geometry": geometry,
        "doping": doping,
        "physics_models": physics_models,
        "mesh": mesh,
        "bias": bias,
    }


if __name__ == "__main__":
    print(json.dumps(run_public_pn_deck("multi_sweep_lifetime", deck_config()), sort_keys=True))
