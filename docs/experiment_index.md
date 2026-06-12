# Experiment Index

`tcad_agent.experiment_index` builds a SQLite index over checkpointed TCAD runs.

It scans:

- `task_run_state.json`
- `sweep_state.json`
- `optimization_state.json`
- agent tool `state.json`
- benchmark `benchmark.json`
- engineering objective `engineering_objectives.json`

The index stores experiment id, kind, status, state path, best objective value, best axis value, quality status, and failure reason.

Known kinds include:

- `task_run`
- `parameter_sweep`
- `adaptive_optimization`
- `multidim_optimization`
- `pn_junction_iv_sweep`
- `diode_breakdown_leakage_sweep`
- `mos_capacitor_cv_sweep`
- `mesh_convergence`
- `tool_convergence`
- `mosfet_2d_id_sweep`
- `extended_device_sweep`
- `schottky_iv_calibration`
- `physical_benchmark`
- `engineering_objective_evaluation`

## Rebuild

```bash
python3.11 -m tcad_agent.experiment_index \
  --rebuild \
  --root runs
```

Default database:

```text
runs/experiment_index.sqlite
```

## List Recent Experiments

```bash
python3.11 -m tcad_agent.experiment_index \
  --list \
  --limit 20
```

## Filter

```bash
python3.11 -m tcad_agent.experiment_index \
  --list \
  --kind adaptive_optimization \
  --status completed
```

This is the first persistent memory layer for a long-running TCAD agent. The next orchestration layer can use it to avoid duplicate runs, retrieve prior best cases, inspect failures, and resume old work.
