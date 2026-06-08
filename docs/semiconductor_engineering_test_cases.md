# Semiconductor TCAD Engineering Test Cases

These cases are the canonical examples for the long-running TCAD agent UI and backend. They now align with the seven public TCAD source categories in `tcad_agent.public_sources`.

Each case should show planning, tool execution or a clear capability boundary, process logs, generated artifacts when executable, quality checks, and an engineering conclusion in the web transcript.

## 1. MOSCAP C-V Oxide QC

Category: MOS capacitor / capacitance

Goal: run MOS capacitor C-V for a process-control wafer. Use P-substrate 1e17 cm^-3, tox 5 nm, and gate sweep -2 V to 2 V with 0.25 V step.

Expected outputs:

- C-V curve image and CSV data;
- extracted Cox, Cmin, and flat-band trend;
- unit, curve-shape, and physical-quality checks;
- conclusion on oxide thickness or fixed-charge abnormality.

## 2. MOSCAP Customer Flat-Band Shift

Category: MOS capacitor / capacitance

Goal: explain a customer-reported MOSCAP flat-band shift near -0.1 V using tox 5 nm and fixed oxide charge 5e11 cm^-2.

Expected outputs:

- C-V result or calibrated planning trace;
- equivalent flat-band shift estimate;
- fixed-charge plausibility check;
- calibration recommendation.

## 3. MOSFET Id-Vg / DIBL Split

Category: MOSFET Id-Vg / Id-Vd / DIBL

Goal: run 2D NMOS Id-Vg at Vd = 0.05 V and 1.0 V, with Vg from 0 V to 1.2 V, and extract Vth shift for DIBL review.

Expected outputs:

- low/high drain Id-Vg curves;
- Vth, SS, Ion/Ioff, and DIBL-risk summary;
- retry log when gate-step refinement or checkpoint reuse is needed;
- final engineering signoff or next experiment suggestion.

## 4. MOSFET Id-Vd Output Kink

Category: MOSFET Id-Vg / Id-Vd / DIBL

Goal: run output characteristics for Vg = 0.8 V, 1.0 V, and 1.2 V, sweeping Vd from 0 V to 1.2 V.

Expected outputs:

- Id-Vd curves and CSV data;
- Ron, saturation-current, and kink/anomaly checks;
- boundary-condition and mesh sanity report;
- conclusion on whether the output characteristic is credible.

## 5. MOSFET Evidence Signoff

Category: MOSFET Id-Vg / Id-Vd / DIBL

Goal: run a primary MOSFET simulation and a mesh/model convergence signoff, with automatic replanning when optional convergence cases fail.

Expected outputs:

- primary curves;
- convergence status;
- agent replan record;
- signoff conclusion.

## 6. Diode / SBD Breakdown And Leakage

Category: Diode / SBD breakdown

Goal: run diode or SBD reverse leakage and breakdown from 0 V to -30 V, starting with 0.5 V bias step and target current threshold 1e-6 A.

Expected outputs:

- reverse IV curve;
- leakage at -5 V and breakdown voltage;
- convergence failures and repair attempts, including reduced bias-step or continuation-ramp retries;
- quality report explaining whether the BV extraction is physically trustworthy.

## 7. Schottky / SBD Barrier Calibration

Category: Diode / SBD breakdown

Goal: calibrate Schottky/SBD barrier height and series resistance against a golden IV curve.

Expected outputs:

- simulated and reference-fit curve;
- barrier height, ideality factor, and log-current RMSE;
- ranked best parameters;
- abnormal points and suggested next calibration round.

## 8. LDMOS BV/Ron Tradeoff

Category: LDMOS / IGBT power devices

Goal: organize a power MOSFET/LDMOS BV and specific-Ron tradeoff baseline, then identify what must be promoted from compact planning to a real high-voltage TCAD runner.

Expected outputs:

- BV/Ron metric table;
- compact-baseline warning;
- high-voltage continuation strategy;
- runner promotion checklist.

## 9. IGBT Turn-Off Tail Template

Category: LDMOS / IGBT power devices

Goal: define the IGBT output, blocking, lifetime, and turn-off tail-current workflow needed for future transient TCAD automation.

Expected outputs:

- IGBT metric list;
- transient solver and layered-geometry gaps;
- DC-to-transient initialization strategy;
- implementation checklist.

## 10. GaN HEMT Output / BV

Category: GaN / AlGaN HEMT

Goal: map GaN HEMT Id-Vg, Id-Vd, 2DEG density, and BV tasks to required polarization, trap, self-heating, and high-field models.

Expected outputs:

- HEMT metric table;
- model-coupling gaps;
- high-field convergence playbook;
- signoff boundary.

## 11. GaN HEMT Current Collapse

Category: GaN / AlGaN HEMT

Goal: draft a stress/recovery experiment plan for current-collapse risk, dynamic Ron ratio, trap occupancy, and gate-edge high-field sensitivity.

Expected outputs:

- off-state stress and recovery plan;
- dynamic Ron metric definition;
- trap/current-collapse model gaps;
- planned-only risk conclusion.

## 12. BJT Gummel / Beta

Category: BJT Gummel / output

Goal: run the BJT `physics_1d` Gummel/output workflow, then prepare correlation against public DEVSIM BJT example sources.

Expected outputs:

- Gummel and output-curve task plan;
- beta, Early voltage, and leakage extraction list;
- base/emitter and collector-bias continuation strategy;
- public-source runner promotion steps.

## 13. BJT Output / Early Voltage

Category: BJT Gummel / output

Goal: plan fixed-Vbe output families, Vce sweeps, collector leakage windows, and Early-voltage extraction with saved bias states.

Expected outputs:

- output-family plan;
- Early-voltage extraction rule;
- collector-leakage sanity window;
- physics_1d evidence boundary and golden/measured correlation recommendation.

## 14. FinFET / GAA DIBL-CV

Category: FinFET / SOI variability

Goal: define FinFET/GAA Id-Vg, Id-Vd, Cgg/Cgd, DIBL, and quantum-correction evidence needed before short-channel signoff.

Expected outputs:

- FinFET metric list;
- 3D geometry and density-gradient gaps;
- DIBL/CV extraction plan;
- signoff boundary.

## 15. SOI / FinFET Variability Campaign

Category: FinFET / SOI variability

Goal: design a nominal-first variability campaign for SOI/FinFET Vth distribution, random trap or geometry splits, mesh reuse, and distribution-level signoff.

Expected outputs:

- sample plan;
- distribution metrics;
- mesh/cache strategy;
- risk conclusion that avoids single-point signoff.

## Coverage Checklist

- MOSFET/DIBL: cases 3, 4, 5
- MOSCAP/capacitance: cases 1, 2
- Diode/SBD breakdown: cases 6, 7
- LDMOS/IGBT: cases 8, 9
- GaN HEMT: cases 10, 11
- BJT: cases 12, 13
- FinFET/SOI variability: cases 14, 15
