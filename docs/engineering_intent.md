# Engineering Intent

`tcad_agent.engineering_intent` turns natural-language TCAD requests into a compact, durable engineering intent record.

It extracts:

- device family and template id;
- executable status: executable, compact baseline, planned, or unknown;
- analyses such as Id-Vg, Id-Vd, C-V, leakage, breakdown, calibration, optimization, and convergence;
- requested metrics such as Vth, SS, Ion/Ioff, DIBL, BV, leakage, ideality factor, Cox, flat-band shift, Ron, and responsivity;
- evidence requirements such as mesh convergence, model A/B, unit check, curve-shape check, golden/measured comparison, and engineering signoff;
- repair preferences such as automatic retry, continuation ramp, and mesh refinement;
- bias and temperature hints found in the user's text.

The mission agent stores this record in `checkpoint.engineering_intent`. Goal decomposition, long-horizon policy, repair planning, and conclusion generation can then use the same interpretation instead of each layer guessing from raw text again.

Planned industrial templates, such as FinFET/GAA, SiC power diode, GaN HEMT, and IGBT, are intentionally routed as high-risk planned capabilities until a runner, quality rules, and benchmark evidence are implemented.
