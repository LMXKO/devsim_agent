# TCAD Repair Strategy

`tcad_agent.repair_strategy` turns a completed or failed run state into a TCAD-specific repair plan.

It does not blindly rerun. It reads:

- `tool_name`;
- original request;
- attempt `failure_class`;
- `quality_report.status`;
- quality issue codes;
- checkpoint hints.

Then it writes:

```text
repair_plan.json
```

## Command

```bash
python3.11 -m tcad_agent.repair_strategy \
  --state runs/agent_tools/pn_junction_iv/suspicious_run/state.json
```

## Planned Repair Actions

The planner can recommend:

- `schema_field_alias_normalization`: map human or LLM aliases such as `output_characteristic`, `transfer_characteristic`, `gate_values`, `vg_start`, and `voltage_start` to executable request fields;
- `continuation_bias_ramp`: shrink bias step and increase attempts after convergence failure;
- `mesh_relax_for_initial_solution`: relax mesh to obtain an initial solution, then verify later;
- `mesh_refinement_and_convergence_check`: refine mesh and run convergence checks when metrics are mesh-sensitive;
- `solver_parameter_adjustment`: increase solver iterations or enable damping where supported;
- `model_switch_staging`: solve Poisson first, then drift-diffusion, then optional advanced models;
- `geometry_sanity_repair`: fix junction/contact/source/drain geometry that violates device bounds;
- `unit_and_bias_range_repair`: narrow suspicious voltage ranges and require unit confirmation;
- `doping_and_unit_sanity_review`: confirm cm^-3, nm, and um units before optimizing;
- `local_bias_step_refinement`: rerun suspicious curve segments at finer bias resolution;
- `mosfet_sweep_range_extension`: extend Id-Vg range when Vth/Ion/Ioff extraction is underconstrained;
- `promote_compact_baseline_to_tcad_runner`: require user confirmation before treating compact baseline evidence as signoff-ready work;
- `implement_planned_industrial_runner_first`: block planned industrial templates until the runner, quality rules, and benchmark path are implemented.

Each action includes:

- priority;
- reason;
- request patch;
- checklist;
- expected effect;
- whether user confirmation is required.

For MOSFET runs, solver repair patches are executable request fields such as `solver_max_iterations`, `solver_relative_error`, `solver_absolute_error`, and `solver_initial_absolute_error`. Mesh repair patches also adjust `x_divisions` and `silicon_y_divisions` when those fields are present.

For validation failures, the schema repair path now preserves the user's physical intent instead of replacing the request with an empty patch. For example, MOSFET `output_characteristic` is normalized to `sweep_type=idvd`, `transfer_characteristic` to `sweep_type=idvg`, and `gate_values` to `idvd_gate_voltage` when possible.

For capability-boundary failures, repair deliberately does not fake a numeric rerun. Compact baseline warnings produce an implementation/correlation checklist, while planned industrial template warnings produce a runner-first checklist.

## Supervisor Integration

The supervisor routes goals containing `repair`, `修复`, `失败恢复`, or `收敛失败` to this tool using the most recent indexed state.

```bash
python3.11 -m tcad_agent.tools.supervisor \
  --goal "给最近失败的 MOSFET DIBL split 或高 Vd Id-Vg 收敛失败生成修复策略" \
  --execute \
  --max-cycles 3
```
