# Semiconductor TCAD Engineering Test Cases

These cases are realistic mission templates for validating the long-running TCAD agent UI and backend. Each case should show planning, tool execution, process logs, generated curves/data artifacts, quality checks, and an engineering conclusion in the web transcript.

## 1. MOS C-V Oxide QC

Goal: run MOS capacitor C-V for a process-control wafer. Use P-substrate 1e17 cm^-3, tox 5 nm, and gate sweep -2 V to 2 V with 0.25 V step.

Expected outputs:

- C-V curve image and CSV data;
- extracted Cox, Cmin, and flat-band trend;
- unit, curve-shape, and physical-quality checks;
- conclusion on oxide thickness or fixed-charge abnormality.

## 2. 2D MOSFET Id-Vg Split

Goal: run 2D NMOS Id-Vg at Vd = 0.05 V and 1.0 V, with Vg from 0 V to 1.2 V at 0.05 V step.

Expected outputs:

- linear and saturation Id-Vg curves;
- Vth, SS, Ion/Ioff, and DIBL-risk summary;
- retry log when gate-step refinement or checkpoint reuse is needed;
- final engineering signoff or next experiment suggestion.

## 3. 2D MOSFET Id-Vd Output

Goal: run output characteristics for Vg = 0.8 V, 1.0 V, and 1.2 V, sweeping Vd from 0 V to 1.2 V at 0.05 V step.

Expected outputs:

- Id-Vd curves and CSV data;
- Ron, saturation-current, and kink/anomaly checks;
- boundary-condition and mesh sanity report;
- conclusion on whether the output characteristic is credible.

## 4. Diode Breakdown And Leakage

Goal: run PN diode reverse leakage and breakdown from 0 V to -30 V, starting with 0.5 V bias step and target current threshold 1e-6 A.

Expected outputs:

- reverse IV curve;
- leakage at -5 V and breakdown voltage;
- convergence failures and repair attempts, including reduced bias-step or continuation-ramp retries;
- quality report explaining whether the BV extraction is physically trustworthy.

## 5. Schottky Barrier Calibration

Goal: calibrate Schottky diode barrier height and series resistance against a golden IV curve.

Expected outputs:

- simulated and reference-fit curve;
- barrier height, ideality factor, and log-current RMSE;
- ranked best parameters;
- abnormal points and suggested next calibration round.

## 6. Mesh Convergence Signoff

Goal: run coarse, nominal, and fine mesh variants for a selected PN IV or MOSFET Id-Vg metric.

Expected outputs:

- curve comparison across mesh levels;
- relative metric delta;
- pass/fail signoff against a 5% stability target;
- mesh refinement recommendation when the metric is not stable.

## 7. Ion/Leakage Pareto Optimization

Goal: run a constrained multi-parameter optimization where Ion/Ioff must be at least 1e4 and reverse leakage is minimized.

Expected outputs:

- 2D sweep heatmap or Pareto plot;
- top candidate process points;
- constraint status for each candidate;
- next adaptive-sampling plan.

## 8. MOS C-V Fixed-Charge Debug

Goal: debug a MOSCAP C-V curve shifted toward negative gate bias by comparing fixed oxide charge assumptions.

Expected outputs:

- C-V overlay;
- flat-band shift estimate;
- fixed-charge diagnosis;
- Chinese engineering conclusion.

## 9. MOSFET Output Kink Debug

Goal: run NMOS Id-Vd output curves for a customer-reported high-Vd kink, allowing natural wording such as "output characteristic".

Expected outputs:

- Id-Vd curves;
- field-alias repair trace when planner wording differs from tool schema;
- Ron/current/kink checks;
- conclusion with mesh/boundary-condition risk.

## 10. High-Temperature Diode Leakage Triage

Goal: use a 300 K reverse-IV baseline to estimate whether a PN diode needs a temperature split for leakage risk.

Expected outputs:

- reverse IV curve;
- leakage at the target reverse bias;
- retry or repair trace;
- next temperature-split recommendation.

## 11. Schottky Golden-Curve Mismatch

Goal: recalibrate Schottky barrier height, ideality factor, and series resistance when the golden-curve residual is high.

Expected outputs:

- residual/fitted curve;
- ranked calibrated parameters;
- physical-quality warnings;
- next scan range.

## 12. PN IV Unit Sanity

Goal: verify whether a suspicious PN IV current magnitude looks like a unit/area mistake or a physical-model issue.

Expected outputs:

- forward IV curve;
- ideality factor and monotonicity check;
- unit sanity diagnosis;
- minimal next experiment.

## 13. MOSFET Vth Shift Triage

Goal: investigate a high-Vth NMOS lot using a quick 2D Id-Vg run and threshold-crossing checks.

Expected outputs:

- Id-Vg curve;
- Vth/SS/Ion-Ioff metrics;
- warning when threshold is not crossed;
- bias/model follow-up advice.

## 14. Mesh/Model Signoff

Goal: run a primary MOSFET simulation and a mesh/model convergence signoff, with automatic replanning when optional convergence cases fail.

Expected outputs:

- primary curves;
- convergence status;
- agent replan record;
- signoff conclusion.

## 15. Existing Bad-Run Repair

Goal: inspect the latest suspicious TCAD run, diagnose mesh/bias/solver/schema problems, and propose or execute the smallest repair.

Expected outputs:

- history lookup;
- repair diagnosis;
- Chinese failure-chain summary;
- minimal next experiment.

## 16-25. Additional Natural-Language Robustness Cases

The web workbench also includes newer cases that intentionally use more human engineering language rather than strict CLI-style fields:

- MOSFET DIBL split review: Id-Vg at low/high drain bias, Vth shift, DIBL risk, and automatic sweep-range repair when threshold is not crossed.
- MOSCAP tox/Qf corner review: oxide-thickness vs fixed-charge diagnosis, Cox comparison, and equivalent flat-band shift.
- Diode BV spec signoff: BV/leakage against a project spec, with reverse-range recommendations when breakdown is not reached.
- PN doping/unit regression: current magnitude, ideality factor, junction/mesh sanity, and likely unit-error diagnosis.
- MOSFET interface-trap SS review: interface trap intent, SS extraction, and model-coupling warnings when physics is metadata-only.
- MOSFET mobility-model A/B: constant vs doping-dependent mobility comparison and model-convergence risk.
- Diode lifetime leakage calibration: reverse leakage triage and next lifetime sweep range.
- Schottky temperature corner: golden-curve calibration plus temperature-extension risk.
- Latest suspicious run explain: history lookup and Chinese failure-chain/repair/risk summary.
- MOSCAP flat-band customer curve: fixed-charge voltage-shift plausibility against a customer-reported flat-band offset.
