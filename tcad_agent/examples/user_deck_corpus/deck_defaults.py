from __future__ import annotations


DEFAULT_GEOMETRY = {
    "length_um": 0.1,
    "junction_um": 0.05,
}

DEFAULT_DOPING = {
    "p_doping_cm3": 1.0e18,
    "n_doping_cm3": 1.0e18,
}

DEFAULT_PHYSICS_MODELS = {
    "electron_lifetime_s": 1.0e-8,
    "hole_lifetime_s": 1.0e-8,
    "temperature_k": 300.0,
}

DEFAULT_MESH = {
    "contact_spacing_um": 0.001,
    "junction_spacing_um": 1.0e-5,
}

