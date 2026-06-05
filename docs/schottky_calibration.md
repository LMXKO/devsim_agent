# Schottky IV Calibration

`tcad_agent.tools.schottky_calibration` calibrates Schottky IV parameters against a trusted curve.

It searches a parameter grid for:

- barrier height
- ideality factor
- series resistance
- image-force barrier lowering

The objective is log-current RMSE in decades. This is intentionally fast: the grid search uses the same thermionic-emission compact current model as the Schottky runner, then optionally verifies the best point with the residual-coupled DEVSIM Schottky path.

Run with the built-in trusted curve:

```bash
python3.11 -m tcad_agent.tools.schottky_calibration \
  --calibration-id schottky_cal_smoke
```

Run against a CSV target:

```bash
python3.11 -m tcad_agent.tools.schottky_calibration \
  --target-curve trusted_schottky_iv.csv \
  --voltage-column voltage_v \
  --current-column current_a \
  --barrier-values 0.68,0.70,0.72,0.74 \
  --ideality-values 1.0,1.08,1.15 \
  --series-resistance-values 0,5,20 \
  --image-force-lowering-values 0,0.01,0.02
```

Verify the best candidate with DEVSIM:

```bash
python3.11 -m tcad_agent.tools.schottky_calibration \
  --calibration-id schottky_cal_devsim \
  --verify-with-devsim
```

Artifacts:

- `state.json`
- `summary.json`
- `target_curve.csv`
- `candidates.csv`
- optional residual-coupled DEVSIM verification state

The experiment index records these states as `schottky_iv_calibration`, with `best_rmse_log_current_dec` as the objective value.
