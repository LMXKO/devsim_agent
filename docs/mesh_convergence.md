# Mesh Convergence Tool

`tcad_agent.tools.mesh_convergence` runs a checkpointed mesh sensitivity check for a TCAD task.

It uses `parameter_sweep` to evaluate the same base task at multiple mesh settings, then compares the objective value between the two finest completed meshes. This is the first automated guard against results that are numerically completed but mesh-dependent.

## Run A Diode/SBD Mesh Check

```bash
python3.11 -m tcad_agent.tools.mesh_convergence \
  --convergence-id diode_leakage_mesh_check \
  --text "diode/SBD reverse leakage 从 0V 扫到 -5V 步长 0.5V max_attempts 3 max_cycles 2" \
  --axis-path mesh.junction_spacing_um \
  --value 2e-5 \
  --value 1e-5 \
  --value 5e-6 \
  --relative-tolerance 0.05 \
  --execute \
  --no-llm
```

Default objective:

```text
minimize abs(final_quality_report.metrics.final_total_current_a)
```

Change it with:

```bash
--objective-metric final_quality_report.metrics.max_abs_current_a
--direction minimize
--raw-objective
```

## Output

```text
runs/mesh_convergence/<convergence_id>/
  state.json
  base_task.json
  conclusion.md
  sweeps/
    <convergence_id>_mesh_sweep/
      sweep_state.json
      summary.csv
```

`quality_report.metrics` includes:

- `axis_path`
- `finest_mesh_value`
- `previous_mesh_value`
- `finest_objective`
- `previous_objective`
- `relative_delta`
- `relative_tolerance`

The quality status is:

- `passed` when the relative delta is within tolerance;
- `suspicious` when the delta exceeds tolerance or some mesh cases failed;
- `failed` when fewer than two mesh cases complete;
- `planned` for dry runs.

The experiment index records this tool under kind `mesh_convergence`.
