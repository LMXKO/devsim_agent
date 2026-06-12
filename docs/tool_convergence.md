# Tool Convergence

`tcad_agent.tool_convergence` runs convergence checks directly over agent-tool request fields.

It complements PN-specific `mesh_convergence` by supporting MOS C-V, diode breakdown, and 2D MOSFET tool requests.

For agent-generated plans, the request boundary normalizes common human/LLM aliases before execution. For example, MOSFET `sweep_type: "output_characteristic"` is converted to `idvd`, `gate_values` is reduced to `idvd_gate_voltage`, and `mesh_refinement_level` is mapped to executable `x_divisions` values.

## Examples

MOSFET mesh convergence over `x_divisions`:

```bash
python3.11 -m tcad_agent.tool_convergence \
  --convergence-id mosfet_x_mesh \
  --tool mosfet_2d_id_sweep \
  --base-request-json '{"sweep_type":"idvg","gate_start":0,"gate_stop":0.5,"gate_step":0.5,"drain_voltage":0.05}' \
  --axis-path x_divisions \
  --value 8 \
  --value 12 \
  --value 16 \
  --metric-path quality_report.metrics.ion_ioff_ratio \
  --relative-tolerance 0.1 \
  --execute
```

MOSFET model convergence over mobility model:

```bash
python3.11 -m tcad_agent.tool_convergence \
  --convergence-id mosfet_mobility_model \
  --tool mosfet_2d_id_sweep \
  --base-request-json '{"sweep_type":"idvg","gate_start":0,"gate_stop":0.5,"gate_step":0.5}' \
  --axis-path mobility_model \
  --value constant \
  --value doping_dependent \
  --metric-path quality_report.metrics.vth_at_threshold_current_v \
  --execute
```

MOS capacitor mesh convergence over oxide spacing:

```bash
python3.11 -m tcad_agent.tool_convergence \
  --tool mos_capacitor_cv_sweep \
  --base-request-json '{"start":-0.5,"stop":0.5,"step":0.25}' \
  --axis-path oxide_spacing_nm \
  --value 0.5 \
  --value 0.25 \
  --value 0.125 \
  --metric-path quality_report.metrics.final_capacitance_f_per_cm2 \
  --execute
```

## Quality Report

The state is written to:

```text
runs/tool_convergence/<convergence_id>/state.json
```

`quality_report.metrics.relative_delta` compares the last two completed cases. If it exceeds `--relative-tolerance`, the convergence result is `suspicious`.
