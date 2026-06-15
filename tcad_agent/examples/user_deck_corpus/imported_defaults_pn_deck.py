from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("ACTSOFT_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcad_agent.examples.user_deck_corpus.common import run_public_pn_deck
from tcad_agent.examples.user_deck_corpus.deck_defaults import (
    DEFAULT_DOPING,
    DEFAULT_GEOMETRY,
    DEFAULT_MESH,
    DEFAULT_PHYSICS_MODELS,
)


geometry = {
    **DEFAULT_GEOMETRY,
    "length_um": 0.12,
}

doping = {
    **DEFAULT_DOPING,
    "n_doping_cm3": 1.2e18,
}

physics_models = {
    **DEFAULT_PHYSICS_MODELS,
    "temperature_k": 310.0,
}

mesh = {
    **DEFAULT_MESH,
}

bias = {
    "name": "imported_forward_iv",
    "start": 0.0,
    "stop": 0.2,
    "step": 0.1,
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
    print(json.dumps(run_public_pn_deck("imported_defaults_pn", deck_config()), sort_keys=True))
