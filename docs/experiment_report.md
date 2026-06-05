# Experiment Report

`tcad_agent.tools.experiment_report` generates a Markdown report from a finished or planned TCAD sweep/optimization state.

It supports:

- `runs/sweeps/<sweep_id>/sweep_state.json`
- `runs/optimizations/<optimize_id>/optimization_state.json`
- the containing sweep or optimization directory

## Generate From An Optimization

```bash
python3.11 -m tcad_agent.tools.experiment_report \
  --state runs/optimizations/p_doping_opt_smoke
```

Default output:

```text
runs/optimizations/p_doping_opt_smoke/report.md
```

## Generate From A Sweep

```bash
python3.11 -m tcad_agent.tools.experiment_report \
  --state runs/sweeps/p_doping_auto_smoke3
```

## Generate From A Multi-Dimensional Optimization

```bash
python3.11 -m tcad_agent.tools.experiment_report \
  --state runs/optimizations/pn_2d_opt
```

## Custom Output

```bash
python3.11 -m tcad_agent.tools.experiment_report \
  --state runs/optimizations/p_doping_opt_smoke/optimization_state.json \
  --output runs/optimizations/p_doping_opt_smoke/p_doping_report.md
```

## Report Contents

The report includes:

- run status, objective, axis, and source state link;
- best result with objective value and quality status;
- best IV curve when the final tool state contains a plot artifact;
- artifact links for CSV, plot, Tecplot, and DEVSIM log;
- ranked cases or observations;
- multi-axis parameter values for `multidim_optimizer` observations.
