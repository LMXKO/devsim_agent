# Experiment Dashboard

`tcad_agent.tools.experiment_dashboard` generates a static HTML dashboard from a TCAD sweep, adaptive optimization, or multi-dimensional optimization state.

It supports:

- `runs/sweeps/<sweep_id>/sweep_state.json`
- `runs/optimizations/<optimize_id>/optimization_state.json`
- the containing sweep or optimization directory

The dashboard is a single HTML file and does not require a local server.

## Generate From An Optimization

```bash
python3.11 -m tcad_agent.tools.experiment_dashboard \
  --state runs/optimizations/p_doping_opt_smoke
```

Default output:

```text
runs/optimizations/p_doping_opt_smoke/dashboard.html
```

## Generate From A Sweep

```bash
python3.11 -m tcad_agent.tools.experiment_dashboard \
  --state runs/sweeps/p_doping_auto_smoke3
```

## Custom Output

```bash
python3.11 -m tcad_agent.tools.experiment_dashboard \
  --state runs/optimizations/p_doping_opt_smoke/optimization_state.json \
  --output runs/optimizations/p_doping_opt_smoke/p_doping_dashboard.html
```

## Dashboard Contents

The dashboard includes:

- experiment status, objective, axis, and quality summary;
- objective trend chart generated as inline SVG;
- objective heatmap for two-axis sweeps or multi-dimensional optimizations;
- best IV plot when available;
- links to state, CSV, plot, and DEVSIM log artifacts;
- ranked result table;
- optimization round table for adaptive and multi-dimensional runs.
