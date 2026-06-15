from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("ACTSOFT_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcad_agent.examples.user_deck_corpus.common import run_public_pn_deck


device_deck = {
    "geometry": {
        "length_um": 0.1,
        "junction_um": 0.05,
    },
    "doping": {
        "p_doping_cm3": 1.0e18,
        "n_doping_cm3": 1.0e18,
    },
    "physics_models": {
        "electron_lifetime_s": 1.0e-8,
        "hole_lifetime_s": 1.0e-8,
        "temperature_k": 300.0,
    },
    "mesh": {
        "contact_spacing_um": 0.001,
        "junction_spacing_um": 1.0e-5,
    },
    "bias": {
        "name": "forward_iv",
        "start": 0.0,
        "stop": 0.2,
        "step": 0.1,
    },
}


def normalized_deck() -> dict[str, object]:
    return {
        "geometry": dict(device_deck["geometry"]),
        "doping": dict(device_deck["doping"]),
        "physics_models": dict(device_deck["physics_models"]),
        "mesh": dict(device_deck["mesh"]),
        "bias": dict(device_deck["bias"]),
    }


if __name__ == "__main__":
    print(json.dumps(run_public_pn_deck("function_wrapped_pn", normalized_deck()), sort_keys=True))
