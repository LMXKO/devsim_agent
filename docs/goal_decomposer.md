# Goal Decomposer

`tcad_agent.tools.goal_decomposer` converts a long-horizon natural-language TCAD goal into durable agent steps.

It is the planning layer above `supervisor` and `mission_agent`.

## Deterministic Plan

```bash
python3.11 -m tcad_agent.tools.goal_decomposer \
  --plan-id mosfet_goal \
  --goal "做 MOSFET Id-Vg，并做 mesh convergence，最后给工程结论"
```

The deterministic planner can add:

- history query;
- primary supervisor execution;
- tool convergence study;
- repair executor step;
- user clarification step;
- final engineering conclusion.

For Schottky calibration goals that mention a trusted/measured curve and convergence, the convergence step targets `schottky_iv_calibration` and sweeps the calibration voltage step size. This keeps calibration missions on the calibration path instead of falling back to the generic PN IV convergence baseline.

If a goal is too abstract, for example "让 TCAD 仿真工程师自动完成我的工作" without a device, analysis type, or target metric, the deterministic plan starts with `ask_user` and stores concrete clarification questions.

If a goal matches a device template that is still marked `planned`, the deterministic plan also starts with `ask_user` and includes the template's missing implementation work instead of pretending the device is executable.

If a goal matches a `compact_baseline` template, the plan can continue into `extended_device_sweep`, but the primary step carries `capability_warnings`, `assumptions`, and an evidence policy that says the result is planning evidence only. Schottky diode routes through `extended_device_sweep` with `fidelity=devsim_1d` and is treated separately from compact baselines.

## LLM Plan

```bash
python3.11 -m tcad_agent.tools.goal_decomposer \
  --plan-id mosfet_goal_llm \
  --goal "优化 MOSFET，让漏电低且 Ion/Ioff 达标，失败时自动修复，最后给结论" \
  --use-llm
```

The LLM must return JSON using supported step kinds:

- `run_supervisor`;
- `run_tool_convergence`;
- `run_repair_executor`;
- `generate_conclusion`;
- `query_history`;
- `ask_user`.

Invalid LLM output falls back to the deterministic plan unless `--no-fallback` is set.

## Mission Integration

`mission_agent` stores the decomposition under:

```text
mission_state.checkpoint.goal_decomposition
```

With `tcad_agent.tools.mission_agent --use-llm`, the mission asks the configured OpenAI-compatible model for this same step plan and then executes it through the mission DAG executor. Invalid LLM output falls back to the deterministic plan unless the mission is run with `--no-llm-fallback`.
