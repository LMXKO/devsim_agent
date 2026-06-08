# TCAD Spec

`tcad_agent.tcad_spec` converts natural language into a structured TCAD task specification.

It builds on `engineering_intent` and `device_templates`, then records:

- device family, template id, support state, suggested tool;
- execution profile: `tcad_executable`, `tcad_signoff_candidate`, `compact_planning_baseline`, `runner_implementation_required`, or `needs_clarification`;
- `tcad_fidelity` and `signoff_workflow`;
- analyses, metrics, objectives, and parsed constraints;
- geometry, materials, model hints, and bias hints;
- signoff requirement and measured/golden reference path;
- capability warnings, assumptions, missing inputs, and clarification questions.

This spec is stored in goal-decomposition requests as `tcad_spec`, so supervisor, mission, benchmark, repair, and conclusion layers can share the same interpretation instead of re-parsing raw text.
