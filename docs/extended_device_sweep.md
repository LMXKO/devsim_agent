# Extended Device Sweep

`tcad_agent.tools.extended_device_sweep` covers two different evidence levels that must not be confused:

- Schottky diode with `fidelity=devsim_1d` invokes the DEVSIM-backed 1D thermionic-emission contact path and is an executable TCAD evidence path.
- Power MOSFET/LDMOS with `fidelity=devsim_2d_field_plate` invokes a DEVSIM 2D layout seed and emits layout-sensitive field-plate/BV/Ron evidence with explicit signoff gaps.
- Power MOSFET/LDMOS with `fidelity=physics_1d` invokes the DEVSIM-backed 1D drift/body baseline runner for fast iterations.
- BJT, GaN, SiC, and IGBT physics routes are executable planning/iteration evidence unless their registry entry says a real solver was invoked.

Supported `device_type` values:

- `schottky_diode`
- `bjt_gummel_output`
- `power_mosfet_bv_ron`
- `finfet_id_cv`
- `sic_power_diode_bv_leakage`
- `gan_hemt_id_bv`
- `igbt_output_turnoff`

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

Power MOSFET/LDMOS can run the 2D field-plate path:

```bash
python3.11 -m tcad_agent.tools.extended_device_sweep \
  --device-type power_mosfet_bv_ron \
  --fidelity devsim_2d_field_plate
```

That path records `tcad_solver_invoked`, `devsim_2d_solver_invoked`, `layout_resolved_field_plate`, `field_peak_x_um`, `field_peak_y_um`, inner DEVSIM CSV/Tecplot/log artifacts, and `runner_contract`. It remains conditional until mesh convergence, calibrated process geometry, and golden/measured correlation pass.

Each state and summary records `evidence_level`. Compact baseline requests also carry `requires_higher_fidelity_runner_for_signoff` and optional `capability_warnings`, so the mission, benchmark, repair, and conclusion layers can keep the result conditional.

The compact runs remain useful for routing, queue validation, benchmark wiring, engineering objective evaluation, and report generation. The `device_templates` catalog records the higher-fidelity DEVSIM implementation steps still needed for each compact or planned device family.
