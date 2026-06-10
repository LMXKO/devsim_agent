# Golden / Measured Curve Comparison

`tcad_agent.tools.golden_curve` compares a TCAD state curve against a trusted or measured reference curve.

The comparison now supports:

- CSV or JSON curve input;
- source/reference x-y column inference or explicit column names;
- automatic unit normalization from column names such as `current_mA`, `current_uA`, `voltage_mV`, and `voltage_kV`;
- optional explicit scale factors for source/reference x and y columns;
- interpolation onto the reference x-grid when points do not exactly match;
- log-domain RMSE/MAE/max-error metrics and sign-mismatch checks;
- a simple current/contact-area scale fit that reports `source_to_reference_y_scale`;
- `aligned_points.csv` and `calibration.json` artifacts for downstream agent inspection.

Example:

```bash
python3.11 -m tcad_agent.tools.golden_curve \
  --source-state runs/example/state.json \
  --reference-curve measured_iv.csv \
  --match-mode interpolate
```

The output state exposes `quality_report.calibration` and `final_summary.calibration`. The physical benchmark treats a completed comparison as golden/measured evidence, while suspicious or failed RMSE still blocks strong signoff.
