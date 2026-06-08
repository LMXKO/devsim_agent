# Autonomous DEVSIM Agent

`tcad_agent.autonomous_devsim_agent` is the direct long-running runtime for the project goal:

> AI autonomously operates DEVSIM-backed TCAD tools over many steps until it produces useful engineering evidence, asks for a required confirmation, or exhausts its budget.

It is broader than the older PN-only `tools/autonomous_loop.py`. The old loop still exists as a narrow checkpointed runner for legacy PN tasks. The new runtime is a tool-using agent shell that can call registered TCAD tools, repair executor, physical benchmark, objective/Pareto evaluator, deck IR/semantic patch utilities, report/conclusion generation, and dashboard generation.

## Loop Shape

```text
goal
  -> observe latest state / metrics / quality / artifacts / logs / curve shape / deck diff
  -> choose one tool action
  -> execute the tool
  -> persist step result and checkpoint
  -> repeat until completed, waiting_for_user, failed, or max_steps
```

Supported actions:

- `audit_capability`: route the goal through the device-template capability catalog and record executable/fidelity/signoff gaps;
- `run_supervisor`: route a natural-language goal into existing supported tools;
- `run_tool`: run a registered TCAD runner such as `pn_junction_iv_sweep`, `mos_capacitor_cv_sweep`, `mosfet_2d_id_sweep`, `diode_breakdown_leakage_sweep`, or `extended_device_sweep`;
- `run_repair_executor`: repair a failed or suspicious state with the repair agent policy and deterministic fallback;
- `run_physical_benchmark`: gate physics, capability boundary, convergence, and measured/golden evidence;
- `evaluate_objectives`: evaluate objectives, constraints, best candidate, and Pareto front before continuing;
- `ingest_deck`: parse a user DEVSIM Python deck into source IR;
- `apply_deck_patch`: apply semantic deck patches and emit patched source plus unified diff;
- `generate_report`: create a sweep/optimization report, or fall back to an engineering conclusion for single-run states;
- `generate_dashboard`: create a dashboard for a sweep, optimization, or autonomous timeline;
- `stop_success`: finish when enough evidence and artifacts exist;
- `ask_user`: stop for high-risk confirmation.

When an OpenAI-compatible model supports native tool/function calling, the agent exposes the dynamic runner registry as tool schemas. If tool calling is unavailable, it falls back to the structured JSON action protocol.

## CLI

```bash
python3.11 -m tcad_agent.tools.autonomous_devsim_agent \
  --goal "自主跑 PN IV，发现曲线或收敛问题就修复，最后给工程结论" \
  --initial-tool-name pn_junction_iv_sweep \
  --initial-request-json '{"start":0,"stop":0.5,"step":0.1,"run_id":"pn_auto_001"}' \
  --execute
```

Run with a user deck, semantic patch, objective gate, and heartbeat/cancel files:

```bash
python3.11 -m tcad_agent.tools.autonomous_devsim_agent \
  --goal "读取我的 DEVSIM deck，调薄 oxide 后跑 IV，并检查漏电/Ron tradeoff" \
  --source-deck-path path/to/user_deck.py \
  --deck-patches-json '[{"deck_path":"geometry.oxide_thickness_nm","request_path":"oxide_thickness_nm","value":45}]' \
  --objectives-json '[{"metric_path":"leakage_current_a","direction":"minimize"}]' \
  --initial-tool-name pn_junction_iv_sweep \
  --execute \
  --allow-user-confirmation-actions
```

Useful options:

- `--max-steps`: total autonomous tool steps before the agent fails closed;
- `--no-llm`: use deterministic policy only;
- `--no-llm-fallback`: fail instead of falling back when the model action is invalid;
- `--allow-user-confirmation-actions`: allow high-risk actions that would otherwise pause;
- `--source-state-path`: resume from an existing TCAD state;
- `--source-deck-path` and `--deck-patches-json`: parse and patch a user deck before running tools;
- `--objectives-json` and `--constraints-json`: add objective/Pareto gates to the loop;
- `--cancel-file` and `--heartbeat-path`: cooperate with external long-run controls;
- `--require-capability-audit`: record executable/fidelity/signoff coverage before running;
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
- queue cancel writes an agent cancel token; the agent checks it at step boundaries and writes heartbeat state;
- queued confirmation pauses can be approved or rejected through the web API;
- deterministic fallback remains available unless disabled;
- compact/planned evidence is still blocked by physical benchmark and signoff evidence gates.
