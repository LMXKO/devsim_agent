# Engineering Intent

`tcad_agent.engineering_intent` turns natural-language TCAD requests into a compact, durable engineering intent record.

It extracts:

- device family and template id;
- executable status: executable, compact baseline, planned, or unknown;
- analyses such as Id-Vg, Id-Vd, C-V, leakage, breakdown, calibration, optimization, and convergence;
- requested metrics such as Vth, SS, Ion/Ioff, DIBL, BV, leakage, ideality factor, Cox, flat-band shift, Ron, and responsivity;
- simple spec constraints such as leakage limits, Ion/Ioff minimums, and BV minimums when they are written in natural language;
- evidence requirements such as mesh convergence, model A/B, unit check, curve-shape check, golden/measured comparison, and engineering signoff;
- repair preferences such as automatic retry, continuation ramp, and mesh refinement;
- bias and temperature hints found in the user's text.

The mission agent stores this record in `checkpoint.engineering_intent`. Goal decomposition, long-horizon policy, repair planning, and conclusion generation can then use the same interpretation instead of each layer guessing from raw text again.

`tcad_agent.tcad_spec` builds a richer `TCADSpec` from this intent. It adds extracted geometry, materials, model hints, bias hints, constraints, signoff requirement, measured/golden reference path, missing inputs, `tcad_fidelity`, and the device template's `signoff_workflow`.

The intent record also carries execution-facing policy fields:

- `clarification_questions`: questions to ask before execution when the request is too abstract or lacks device, analysis, and metric intent.
- `assumptions`: explicit assumptions that downstream agents must preserve in checkpoints and conclusions.
- `capability_warnings`: warnings when a goal maps to a compact baseline or a planned template.
- `evidence_policy`: one of `executable_exploratory`, `requires_signoff_evidence`, `compact_planning_only`, `blocked_until_runner_implemented`, or `needs_clarification`.

Capability boundaries are intentionally conservative:

- executable templates, such as PN IV, MOS C-V, diode leakage/BV, 2D MOSFET Id sweeps, and Schottky `devsim_1d`, may run automatically as engineering evidence;
- compact baseline templates in the seven-category showcase, such as BJT and power MOSFET/LDMOS, may run automatically only as planning evidence and must not be promoted to signoff conclusions;
- planned industrial templates, such as FinFET/GAA, SiC power diode, GaN HEMT, and IGBT, are blocked until a runner, quality rules, and benchmark evidence are implemented.
