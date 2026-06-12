# Engineering Objectives

`tcad_agent.engineering_objectives` evaluates engineering constraints and Pareto objectives over TCAD result states.

It can read a single tool `state.json` or an aggregate `optimization_state.json` / `sweep_state.json`.

Example:

```bash
python3.11 -m tcad_agent.engineering_objectives \
  --state runs/optimizations/mosfet_opt/optimization_state.json \
  --objective ion_ioff_ratio:maximize \
  --constraint 'ioff_current_a<=1e-10'
```

Power-device tradeoff example:

```bash
python3.11 -m tcad_agent.engineering_objectives \
  --state runs/optimizations/power_mos_opt/optimization_state.json \
  --objective breakdown_voltage_v:maximize_abs \
  --objective specific_on_resistance_ohm_cm2:minimize
```

The tool writes `engineering_objectives.json` with:

- evaluated candidates;
- constraint violations;
- feasible/infeasible status;
- weighted score;
- Pareto-front flag;
- best feasible candidate;
- `decision`, a machine-readable continue/review/reject/collect-more-evidence summary for the agent loop.

The experiment index records this file as `engineering_objective_evaluation`.
