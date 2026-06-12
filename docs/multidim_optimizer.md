# Multi-Dimensional Optimizer

`tcad_agent.multidim_optimizer` runs checkpointed coarse-to-fine optimization across two or more numeric TCAD task fields.

It reuses `parameter_sweep` as the execution layer. Each candidate point is launched as a one-case sweep, so the optimizer can skip parameter combinations that were already evaluated when a long run is resumed.

## Execute A 2D Optimization

```bash
python3.11 -m tcad_agent.multidim_optimizer \
  --optimize-id diode_bv_2d_opt \
  --text "diode/SBD reverse leakage 从 0V 扫到 -5V 步长 0.5V，优化掺杂和结位置，让漏电最小且 BV 风险可解释，max_attempts 3 max_cycles 2" \
  --axis parameters.p_doping_cm3:log:1e16:1e18:3 \
  --axis parameters.junction_um:linear:0.04:0.06:3 \
  --max-rounds 2 \
  --max-cases-per-round 9 \
  --execute \
  --no-llm
```

Axis syntax:

```text
path:scale:min:max:initial_points[:max_new_points_per_round]
```

Examples:

- `parameters.p_doping_cm3:log:1e16:1e18:3`
- `parameters.junction_um:linear:0.04:0.06:3:2`

By default the objective is:

```text
minimize abs(final_quality_report.metrics.final_total_current_a)
```

Change it with:

```bash
--objective-metric final_quality_report.metrics.final_total_current_a
--direction maximize
--raw-objective
```

## Checkpoint Layout

```text
runs/optimizations/<optimize_id>/
  base_task.json
  optimization_state.json
  rounds/
    <optimize_id>_round_001/
      summary.csv
  sweeps/
    <optimize_id>_round_001_point_001/
      sweep_state.json
      summary.csv
      tasks/
      agent_tools/
```

If the same `--optimize-id` is reused without `--overwrite`, planned observations are discarded when executing and completed observations are kept. The optimizer then adds missing rounds until `--max-rounds` or `--max-cases` is reached.

## Strategy

Round 1 evaluates the Cartesian product of each axis' initial values.

Later rounds:

- choose the best completed observation according to the objective;
- propose midpoints between the best value and its nearest evaluated neighbors on each axis;
- add coordinate refinements that hold other axes at the current best point;
- filter parameter combinations that were already evaluated;
- stop when no new point remains inside the configured bounds.

Use log scale for doping-like quantities and linear scale for geometry, temperature, and voltage-like quantities.

## Reports And Dashboards

The multi-dimensional optimizer works with the normal report tools:

```bash
python3.11 -m tcad_agent.reporting \
  --state runs/optimizations/diode_bv_2d_opt

python3.11 -m tcad_agent.dashboard \
  --state runs/optimizations/diode_bv_2d_opt

python3.11 -m tcad_agent.conclusion \
  --state runs/optimizations/diode_bv_2d_opt
```

For exactly two axes, the dashboard renders an objective heatmap. For three or more axes, use the ranked observations table and continue refinement around the best parameter combination.
