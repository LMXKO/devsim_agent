# 2D MOSFET Id-Vg / Id-Vd Tool

`tcad_agent.tools.mosfet_2d_id` runs a simplified 2D MOSFET-like DEVSIM structure and extracts Id-Vg / Id-Vd metrics.

This is the first 2D device runner in the project. It uses a hand-built Gmsh-like triangular mesh imported into DEVSIM:

- oxide region over silicon;
- oxide/silicon interface;
- gate contact on oxide;
- n+ source/drain regions in a p-type silicon body;
- source, drain, and body contacts on silicon.

The current geometry is intentionally conservative and lightweight for agent smoke tests. It is not yet a calibrated process/device deck. The quality checker may mark runs as `suspicious` when the numerical result is complete but the extracted curves are physically weak.

## Physics Model Options

The runner exposes explicit model configuration:

- `--mobility-model constant|doping_dependent`;
- `--electron-mobility-cm2-v-s`;
- `--hole-mobility-cm2-v-s`;
- `--recombination-model none|srh`;
- `--electron-lifetime-s`;
- `--hole-lifetime-s`;
- `--interface-trap-density-cm2`;
- `--fixed-oxide-charge-cm2`;
- `--impact-ionization-model none|selberherr`;
- `--model-strategy poisson_then_dd|dd_direct`.

Implemented in the equations today:

- constant/effective mobility is written to DEVSIM `mu_n` / `mu_p`;
- SRH lifetime parameters are written to `taun` / `taup`;
- `recombination-model none` approximates disabled SRH by using very long lifetimes;
- Poisson then drift-diffusion staging remains the default solve path.

Metadata-only today:

- interface trap density;
- fixed oxide charge;
- impact ionization.

When metadata-only advanced models are requested, the quality checker marks the result `suspicious` so the agent does not treat it as a fully coupled calibrated deck.

## Run A 2D MOSFET Smoke

```bash
python3.11 -m tcad_agent.tools.mosfet_2d_id \
  --run-id mosfet2d_tool_smoke \
  --sweep-type both \
  --gate-start 0 \
  --gate-stop 0.5 \
  --gate-step 0.5 \
  --drain-voltage 0.05 \
  --drain-start 0 \
  --drain-stop 0.05 \
  --drain-step 0.05 \
  --idvd-gate-voltage 0.5 \
  --x-divisions 8 \
  --silicon-y-divisions 3
```

Output layout:

```text
runs/agent_tools/mosfet_2d_id/<run_id>/
  state.json
  conclusion.md
  attempt_runs/
    mosfet_2d/attempt_001/
      mosfet_id_sweep.csv
      mosfet_id_curves.png
      device_tecplot.dat
      devsim.log
      summary.json
```

## Extracted Metrics

`quality_report.metrics` includes:

- `vth_at_threshold_current_v`
- `subthreshold_swing_mv_dec`
- `ion_current_a`
- `ioff_current_a`
- `ion_ioff_ratio`
- `max_transconductance_s`
- `idvd_final_current_a`
- `output_conductance_last_s`
- `idvg_shape_violations`

## Quality Checks

The tool marks results as suspicious when:

- Id-Vg absolute drain current is non-monotonic;
- Ion/Ioff is below `--quality-min-ion-ioff-ratio`;
- drain current exceeds `--quality-max-abs-current-a`;
- subthreshold swing is below the room-temperature thermal limit or unusually large;
- Id-Vg never crosses the configured threshold current;
- source/drain geometry, oxide thickness, temperature, doping, or mesh dimensions fail sanity checks;
- too few points are available for extraction.

The tool retries convergence failures with smaller gate/drain bias steps and records every attempt in `state.json`.

## Current Limitation

The first 2D MOSFET implementation is a real DEVSIM 2D solve, but it is still a coarse reference deck. Treat it as a runnable agent target for workflow development and regression tests, then refine geometry, contacts, mobility/interface physics, and mesh for engineering-grade MOSFET studies.

The experiment index records this tool under kind `mosfet_2d_id_sweep`.
