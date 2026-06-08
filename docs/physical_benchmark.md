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
- Extended devices: Schottky/SBD barrier/ideality, DEVSIM solver metadata, thermionic contact-model metrics, and residual-coupling confirmation when `fidelity=devsim_1d`; BJT `physics_1d` gain/Early/output-family/collector-leakage evidence; power MOSFET/LDMOS `physics_1d` BV/Ron/field-peak/impact-ionization evidence. JFET and photodiode compact checks remain legacy regression coverage.
- Mesh/model convergence: final relative delta vs tolerance.
- Sweeps/optimizers: completed/suspicious/failed case mix plus best-child state benchmark when available.
- Golden profiles: built-in metric targets for extended-device baselines, plus optional `golden_metrics` overrides from request or metrics.

## Capability Boundary Checks

The benchmark layer now records a capability boundary in `credibility_assessment.evidence_matrix.capability_boundary`.

- `tcad_executable` or equivalent evidence means the tool exposed an executable TCAD path, though signoff may still need convergence and golden/measured comparison.
- `compact_baseline` means the run is only planning evidence. The benchmark adds `compact_baseline_not_signoff_evidence`, and the engineering decision is conditional.
- `planned_runner_missing` means a planned industrial device was represented by a surrogate or compact output. The benchmark adds `planned_industrial_template_runner_missing`, and signoff is blocked until a real runner, quality rules, and benchmark evidence exist.

## Result Meaning

- `passed`: no benchmark warnings or errors.
- `suspicious`: no hard error, but one or more physical checks deserve inspection.
- `failed`: at least one benchmark is physically inconsistent enough to block trust.
- `unsupported`: the state did not expose a supported tool type or quality report.

The experiment index records generated `benchmark.json` files as `physical_benchmark` records, so long-running agents can retrieve benchmark status from history.

`credibility_assessment` also lists `evidence_gaps` and `must_fix_before_signoff`. These fields are the handoff between benchmark, repair planning, and final engineering conclusions.

The summary also includes `signoff_evidence_pack`. This pack gates:

- `quality_report`;
- curve/data artifacts;
- structured TCAD deck/spec;
- physical benchmark;
- convergence evidence when signoff requires it;
- golden/measured comparison when requested;
- capability boundary, including compact baseline and planned-runner blocks.
