# Extended Device Sweep

`tcad_agent.tools.extended_device_sweep` provides executable baselines for device templates that are not yet all full DEVSIM structures.

Supported `device_type` values:

- `schottky_diode`
- `bjt_gummel_output`
- `jfet_transfer_output`
- `power_mosfet_bv_ron`
- `photodiode_iv`

Example:

```bash
python3.11 -m tcad_agent.tools.extended_device_sweep \
  --device-type schottky_diode \
  --run-id schottky_smoke
```

Schottky can opt into a DEVSIM-backed 1D thermionic-emission contact solve:

```bash
python3.11 -m tcad_agent.tools.extended_device_sweep \
  --device-type schottky_diode \
  --fidelity devsim_1d \
  --start -0.1 \
  --stop 0.1 \
  --step 0.1 \
  --schottky-contact-coupling-mode residual \
  --run-id schottky_devsim_smoke
```

Each run writes:

- `state.json`
- `summary.json`
- `sweep.csv`
- `curve.svg`
- `extended_device.log`

For Schottky `fidelity=devsim_1d`, the summary also records `tcad_solver_invoked`, `solver_backend`, `schottky_contact_model`, `schottky_contact_coupling_mode`, `thermionic_residual_coupled`, thermionic contact-current metrics, `devsim_log`, `tecplot`, and the inner DEVSIM summary. Optional Schottky parameters include `schottky_series_resistance_ohm`, `schottky_image_force_lowering_ev`, and `schottky_auto_image_force_lowering`.

The compact runs remain useful for routing, queue validation, benchmark wiring, engineering objective evaluation, and report generation. The `device_templates` catalog records the higher-fidelity DEVSIM implementation steps still needed for each device family.
