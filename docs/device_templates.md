# Device Task Templates

`tcad_agent.tools.device_templates` is the device-routing catalog for long-horizon TCAD missions.

It separates two states that should not be confused:

- `executable`: the project has a runnable tool path today.
- `planned`: the project knows the device/task template, expected metrics, and missing implementation work, but should not pretend it can simulate it yet.

## List Templates

```bash
python3.11 -m tcad_agent.tools.device_templates list
python3.11 -m tcad_agent.tools.device_templates list --support executable
```

## Route A Goal

```bash
python3.11 -m tcad_agent.tools.device_templates route \
  --goal "做 Schottky diode forward IV 并提取 barrier height"
```

The route result includes the matched template, support state, suggested executable tool when available, default request hints, missing capabilities, and next implementation steps.

## Executable Templates

- PN junction IV: `pn_junction_iv_sweep`
- MOS capacitor C-V: `mos_capacitor_cv_sweep`
- diode leakage/breakdown: `diode_breakdown_leakage_sweep`
- 2D MOSFET Id-Vg / Id-Vd: `mosfet_2d_id_sweep`
- Schottky diode IV: `extended_device_sweep` compact by default; `fidelity=devsim_1d` enables the DEVSIM-backed thermionic-emission contact model path.
- BJT Gummel/output compact baseline: `extended_device_sweep`
- JFET transfer/output compact baseline: `extended_device_sweep`
- power MOSFET BV/Ron compact baseline: `extended_device_sweep`
- photodiode dark/illuminated IV compact baseline: `extended_device_sweep`

## Compact Baselines

The extended templates generate deterministic curves, metrics, artifacts, quality reports, benchmark checks, and indexable state. Schottky now has the first DEVSIM-backed thermionic-emission contact model path; BJT, JFET, power MOSFET, and photodiode remain compact baselines. Their `next_implementation_steps` describe the higher-fidelity DEVSIM work still needed before treating them as final industrial TCAD evidence.

`supervisor` now uses this catalog to avoid routing specialized devices into the wrong executable tool. For example, "power MOSFET BV" routes to `extended_device_sweep` with `device_type=power_mosfet_bv_ron` instead of being run as a simple 2D MOSFET transfer curve.

`goal_decomposer` also uses this catalog. If a future device is marked `planned`, it will create a confirmation/implementation step before execution.
