# Autonomous DEVSIM Agent

`tcad_agent.autonomous_devsim_agent` is the direct long-running runtime for the project goal:

> AI autonomously operates DEVSIM-backed TCAD tools over many steps until it produces useful engineering evidence, asks for a required confirmation, or exhausts its budget.

It is broader than the older PN-only `tools/autonomous_loop.py`. The old loop still exists as a narrow checkpointed runner for legacy PN tasks. The new runtime is a tool-using agent shell that can call registered TCAD tools, repair executor, physical benchmark, report/conclusion generation, and dashboard generation.

## Loop Shape

```text
goal
  -> observe latest state / metrics / quality / artifacts
  -> choose one tool action
  -> execute the tool
  -> persist step result and checkpoint
  -> repeat until completed, waiting_for_user, failed, or max_steps
```

Supported actions:

- `run_supervisor`: route a natural-language goal into existing supported tools;
- `run_tool`: run a registered TCAD runner such as `pn_junction_iv_sweep`, `mos_capacitor_cv_sweep`, `mosfet_2d_id_sweep`, `diode_breakdown_leakage_sweep`, or `extended_device_sweep`;
- `run_repair_executor`: repair a failed or suspicious state with the repair agent policy and deterministic fallback;
- `run_physical_benchmark`: gate physics, capability boundary, convergence, and measured/golden evidence;
- `generate_report`: create a sweep/optimization report, or fall back to an engineering conclusion for single-run states;
- `generate_dashboard`: create a dashboard when the latest state is a sweep or optimization;
- `stop_success`: finish when enough evidence and artifacts exist;
- `ask_user`: stop for high-risk confirmation.

## CLI

```bash
python3.11 -m tcad_agent.tools.autonomous_devsim_agent \
  --goal "自主跑 PN IV，发现曲线或收敛问题就修复，最后给工程结论" \
  --initial-tool-name pn_junction_iv_sweep \
  --initial-request-json '{"start":0,"stop":0.5,"step":0.1,"run_id":"pn_auto_001"}' \
  --execute
```

Useful options:

- `--max-steps`: total autonomous tool steps before the agent fails closed;
- `--no-llm`: use deterministic policy only;
- `--no-llm-fallback`: fail instead of falling back when the model action is invalid;
- `--allow-user-confirmation-actions`: allow high-risk actions that would otherwise pause;
- `--source-state-path`: resume from an existing TCAD state;
- `--resume --agent-id ...`: resume an existing agent state.

## Queue Integration

The run queue registers `autonomous_devsim_agent`, so a long job can be enqueued:

```bash
python3.11 -m tcad_agent.tools.run_queue enqueue \
  --tool autonomous_devsim_agent \
  --request-json '{"goal_text":"自主完成 PN IV，失败时修复并给结论","execute":true,"max_steps":8}'
```

## Safety Boundary

The runtime is agent-first, but not unrestricted:

- one action per step;
- no shell commands from the model;
- unsupported tool names are rejected;
- high-risk geometry/process/model edits pause unless confirmation is allowed;
- deterministic fallback remains available unless disabled;
- compact/planned evidence is still blocked by physical benchmark and signoff evidence gates.

