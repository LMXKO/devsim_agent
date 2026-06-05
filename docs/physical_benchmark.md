# Physical Benchmark

`tcad_agent.tools.physical_benchmark` runs a second-layer physical sanity benchmark over a saved TCAD state.

It is complementary to `quality_report`: quality checks answer "did the run produce usable numeric artifacts?", while benchmark checks answer "are the numbers plausible against simple physics or engineering golden ranges?"

## Run

```bash
python3.11 -m tcad_agent.tools.physical_benchmark \
  --state runs/agent_tools/mosfet_2d_id/fet_a/state.json
```

This writes:

```text
runs/agent_tools/mosfet_2d_id/fet_a/benchmark.json
```

## Current Benchmarks

- PN junction IV: ideality factor, rectification ratio, turn-on voltage, thermal voltage reference.
- MOS capacitor C-V: oxide capacitance upper bound, positive capacitance, min/max ordering, C-V dynamic range, and fixed-charge equivalent voltage-shift sanity.
- Diode leakage/breakdown: leakage threshold, reverse breakdown polarity, reverse-current monotonicity.
- 2D MOSFET Id-Vg / Id-Vd: thermal-limit subthreshold swing, Ion/Ioff, Vth inside gate sweep, extracted magnitude signs, Id-Vd negative-differential segments, kink-like slope jumps, and saturation-shape checks.
- Extended devices: Schottky barrier/ideality, DEVSIM solver metadata, thermionic contact-model metrics, and residual-coupling confirmation when `fidelity=devsim_1d`; BJT gain, JFET pinch-off, power MOS BV/Ron, photodiode responsivity.
- Mesh/model convergence: final relative delta vs tolerance.
- Sweeps/optimizers: completed/suspicious/failed case mix plus best-child state benchmark when available.
- Golden profiles: built-in metric targets for compact extended-device baselines, plus optional `golden_metrics` overrides from request or metrics.

## Result Meaning

- `passed`: no benchmark warnings or errors.
- `suspicious`: no hard error, but one or more physical checks deserve inspection.
- `failed`: at least one benchmark is physically inconsistent enough to block trust.
- `unsupported`: the state did not expose a supported tool type or quality report.

The experiment index records generated `benchmark.json` files as `physical_benchmark` records, so long-running agents can retrieve benchmark status from history.
