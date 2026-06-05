# TCAD Metrics

`tcad_agent.metrics` extracts engineering metrics from simulator artifacts.

The first supported extractor is PN junction IV:

- point count and voltage range;
- final, min, max, and max-absolute current;
- current near 0 V;
- leakage-current proxy;
- turn-on voltage at 1 uA;
- estimated ideality factor;
- last-segment differential resistance;
- final-to-leakage rectification ratio;
- reverse breakdown proxy at 1 uA when a reverse sweep is present.
- reverse leakage and breakdown fields when a reverse sweep is present.

The PN runner writes these values into `summary.json` under:

```json
"extracted_metrics": {
  "turn_on_voltage_at_1ua_v": 0.1198,
  "ideality_factor_estimate": 1.51
}
```

`tcad_agent.tools.result_judge` also includes these values in `quality_report.metrics` and uses them for the first physical plausibility checks:

- ideality factor range;
- rectification ratio;
- temperature sanity range;
- voltage-span sanity range;
- doping, geometry, oxide, and mesh-spacing sanity ranges;
- existing numerical checks such as monotonic voltage, finite values, point count, and current limits.

For diode reverse leakage / breakdown tasks, `extract_diode_reverse_metrics` adds:

- `leakage_abs_current_at_target_a`;
- `leakage_voltage_used_v`;
- `max_reverse_abs_current_a`;
- `breakdown_current_threshold_a`;
- `breakdown_voltage_at_threshold_v`;
- `breakdown_detected`;
- `reverse_abs_current_gain`;
- `reverse_current_shape_violations`.

For 2D MOSFET Id sweeps, `extract_mosfet_metrics` adds:

- `vth_at_threshold_current_v`;
- `subthreshold_swing_mv_dec`;
- `ion_current_a`;
- `ioff_current_a`;
- `ion_ioff_ratio`;
- `max_transconductance_s`;
- `idvd_final_current_a`;
- `output_conductance_last_s`.

`tcad_agent.physical_quality` provides reusable checks for:

- oxide capacitance estimates for MOS C-V;
- capacitance values above Cox or outside broad unit sanity ranges;
- MOSFET subthreshold swing below the room-temperature thermal limit;
- missing threshold crossing;
- source/drain geometry and channel sanity.

This layer is intentionally separate from any one simulator runner so future devices can add their own extractors, such as MOS C-V flatband/threshold metrics or MOSFET Id-Vg Vth/SS/Ion/Ioff metrics.
