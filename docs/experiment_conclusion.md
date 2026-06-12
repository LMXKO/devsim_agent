# Experiment Conclusion

`tcad_agent.conclusion` generates a conclusion-oriented Markdown summary from TCAD state.

Unlike `experiment_report`, which focuses on tables and artifact links, the conclusion report focuses on:

- best result;
- objective value;
- trend interpretation and confidence;
- key extracted metrics;
- suspicious/failed cases and outlier anomalies;
- physical benchmark summary;
- recommended next experiment;
- a structured next experiment plan that an agent can turn into follow-up tool calls.

## Generate From An Optimization

```bash
python3.11 -m tcad_agent.conclusion \
  --state runs/optimizations/p_doping_opt_smoke
```

Default output:

```text
runs/optimizations/p_doping_opt_smoke/conclusion.md
```

## Generate From A Multi-Dimensional Optimization

```bash
python3.11 -m tcad_agent.conclusion \
  --state runs/optimizations/pn_2d_opt
```

For multi-axis states, the conclusion report treats the result as a response surface. It reports the best sampled parameter combination and recommends additional local refinement instead of forcing a one-dimensional trend explanation.

## Generate From A Tool State

```bash
python3.11 -m tcad_agent.conclusion \
  --state runs/supervisor/supervisor_mos_smoke2/agent_tools/mos_capacitor_cv/supervisor_mos_smoke2_mos_cv_002/state.json
```

## Current Behavior

The conclusion generator is deterministic. It does not invent physics explanations; it derives conclusions from state files, objective values, quality status, extracted metrics, and physical benchmark checks.

When possible, it also writes or refreshes `benchmark.json` through `physical_benchmark` and includes:

- benchmark status and check counts;
- warning/error benchmark codes;
- an engineering decision: accept as baseline, use conditionally, or do not trust yet;
- signoff evidence pack, including missing evidence, blocking reasons, and required evidence item statuses;
- anomalies such as failed/suspicious cases, missing objectives, objective outliers, and nonmonotonic single-axis trends;
- next experiment packages with a target tool and request hint.
- executable next-request JSON snippets that can seed a follow-up agent/tool call.

An LLM can be layered on top later to turn this structured conclusion into richer prose while keeping the evidence grounded.
