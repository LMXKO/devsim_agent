# Schottky 1D DEVSIM Runner

`tcad_agent.examples.schottky_1d.run` is the first higher-fidelity bridge for the extended device catalog.

It builds a 1D n-type silicon mesh in DEVSIM, applies a metal-side Schottky contact potential and carrier boundary, registers a thermionic-emission contact current model, couples that current into the metal contact electron-continuity residual by default, solves Poisson plus drift-diffusion, and writes raw DEVSIM current columns alongside the area-scaled Schottky IV current used by the agent layer.

Run directly:

```bash
python3.11 -m tcad_agent.examples.schottky_1d.run \
  --start -0.1 \
  --stop 0.1 \
  --step 0.1 \
  --run-id schottky_1d_smoke
```

Run through the agent-callable tool:

```bash
python3.11 -m tcad_agent.tools.extended_device_sweep \
  --device-type schottky_diode \
  --fidelity devsim_1d \
  --start -0.1 \
  --stop 0.1 \
  --step 0.1 \
  --schottky-image-force-lowering-ev 0.01 \
  --schottky-series-resistance-ohm 5 \
  --schottky-contact-coupling-mode residual \
  --run-id schottky_devsim_smoke
```

Artifacts:

- `sweep.csv`
- `summary.json`
- `devsim.log`
- `device_tecplot.dat`
- outer `state.json` when launched through `extended_device_sweep`

The runner records `solver_backend=devsim_1d_thermionic_emission_contact_model`, `schottky_contact_model`, `schottky_contact_coupling_mode`, `thermionic_residual_coupled`, `series_resistance_ohm`, `image_force_lowering_enabled`, and contact electric-field metrics.

Current limitation: the thermionic-emission contact current is now residual-coupled, but final industrial signoff still needs calibrated material parameters, contact-area normalization against a known reference, image-force lowering coupled directly to the solved field inside the contact model, and broader validation against measured or trusted TCAD curves.
