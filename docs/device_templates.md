# Device Task Templates

`tcad_agent.tools.device_templates` is the device-routing catalog for long-horizon TCAD missions.

It separates three states that should not be confused:

- `executable`: the project has a runnable TCAD-backed path that can produce engineering evidence.
- `compact_baseline`: the project can generate deterministic planning curves, metrics, artifacts, and quality reports, but these outputs are not final TCAD signoff evidence.
- `planned`: the project knows the device/task template, expected metrics, and missing implementation work, but should not pretend it can simulate it yet.

## List Templates

```bash
python3.11 -m tcad_agent.tools.device_templates list
python3.11 -m tcad_agent.tools.device_templates list --support executable
python3.11 -m tcad_agent.tools.device_templates list --support compact_baseline
python3.11 -m tcad_agent.tools.device_templates list --support planned
```

## Route A Goal

```bash
python3.11 -m tcad_agent.tools.device_templates route \
  --goal "做 Schottky/SBD forward IV 并提取 barrier height"
```

The route result includes the matched template, support state, suggested executable tool when available, default request hints, missing capabilities, and next implementation steps.
It also includes `tcad_fidelity` and `signoff_workflow`, so downstream agents can tell a real TCAD evidence path from a compact planning route.
For public-source seeding, route results can also include `public_source_category_ids`, `public_sources`, and `recommended_convergence`.

## Public Sources

The seven public source categories are recorded in `tcad_agent.public_sources` and documented in [tcad_public_sources.md](tcad_public_sources.md). They cover:

- MOSFET Id-Vg / Id-Vd / DIBL
- Diode / SBD breakdown
- LDMOS / IGBT power devices
- GaN / AlGaN HEMT
- BJT Gummel / output
- FinFET / SOI variability
- MOS capacitor / capacitance

List the registry:

```bash
python3.11 -m tcad_agent.tools.device_templates sources
python3.11 -m tcad_agent.tools.device_templates sources --kind categories
python3.11 -m tcad_agent.tools.device_templates sources --kind sources
```

## Executable Templates

- PN junction IV: `pn_junction_iv_sweep`
- MOS capacitor C-V: `mos_capacitor_cv_sweep`
- diode leakage/breakdown: `diode_breakdown_leakage_sweep`
- 2D MOSFET Id-Vg / Id-Vd: `mosfet_2d_id_sweep`
- Schottky diode IV: `extended_device_sweep` with `fidelity=devsim_1d` for the DEVSIM-backed thermionic-emission contact model path.
- BJT Gummel/output: `extended_device_sweep` with `fidelity=physics_1d`, including a Gummel sweep, Ic-Vce output family, Early-effect evidence, beta extraction, and collector leakage.
- power MOSFET/LDMOS BV/Ron: `extended_device_sweep` with `fidelity=physics_1d`, including drift-region Ron decomposition, peak-field evidence, and local-field impact-ionization coupling for BV extraction.

These executable templates carry TCAD fidelity labels such as `devsim_1d_drift_diffusion`, `devsim_1d_quasi_static_cv`, `devsim_1d_reverse_iv`, `devsim_2d_drift_diffusion`, `devsim_1d_thermionic_contact`, `physics_1d_bjt_transport`, and `physics_1d_high_voltage_drift_avalanche`. Their signoff workflows require physical benchmark plus convergence and golden/measured comparison when requested.

## Compact Baselines

Legacy JFET and photodiode compact routes remain in code for regression coverage, but they are no longer used as public seven-category examples. Compact templates generate deterministic curves, metrics, artifacts, quality reports, benchmark checks, and indexable state. They are useful for routing, UI validation, planning, and smoke evidence. `physical_benchmark` marks them with `compact_baseline_not_signoff_evidence`, so conclusions are conditional until a higher-fidelity runner or golden/measured correlation is added.

## Planned Industrial Templates

FinFET/GAA, SiC power diode, GaN HEMT, and IGBT are planned industrial templates. They remain blocked at goal decomposition and supervisor routing until a real runner, quality rules, and benchmark evidence are implemented.

`supervisor` now uses this catalog to avoid routing specialized devices into the wrong executable tool. For example, "power MOSFET BV" routes to `extended_device_sweep` with `device_type=power_mosfet_bv_ron` instead of being run as a simple 2D MOSFET transfer curve.

`goal_decomposer` also uses this catalog. If a future device is marked `planned`, it creates a confirmation/implementation step before execution. If a device is `compact_baseline`, it can continue as a planning run but carries capability warnings into the mission state and final conclusion.
