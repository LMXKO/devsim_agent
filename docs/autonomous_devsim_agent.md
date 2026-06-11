# Autonomous DEVSIM Agent

`tcad_agent.autonomous_devsim_agent` is the direct long-running runtime for the project goal:

> AI autonomously operates DEVSIM-backed TCAD tools over many steps until it produces useful engineering evidence, asks for a required confirmation, or exhausts its budget.

It is the project-level agent-first controller. It can call registered TCAD tools, repair executor, physical benchmark, objective/Pareto evaluator, deck IR/semantic patch utilities, Power MOSFET signoff workflow, external Sentaurus runner contracts, report/conclusion generation, and minimal cockpit generation. The checkpoint records `agent_control`, `agent_decision_ledger`, and `agent_hypothesis_tree` so long runs remain auditable.

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
- `run_tool`: run a registered TCAD runner such as `pn_junction_iv_sweep`, `mos_capacitor_cv_sweep`, `mosfet_2d_id_sweep`, `diode_breakdown_leakage_sweep`, `extended_device_sweep`, `power_mosfet_signoff`, `industrial_external_tcad_runner`, or industrial aliases such as `power_mosfet_bv_ron_2d_runner`;
- `run_repair_executor`: repair a failed or suspicious state with the repair agent policy and deterministic fallback;
- `run_physical_benchmark`: gate physics, capability boundary, convergence, and measured/golden evidence;
- `evaluate_objectives`: evaluate objectives, constraints, best candidate, and Pareto front before continuing;
- `ingest_deck`: parse a user DEVSIM Python deck into source IR;
- `apply_deck_patch`: apply semantic deck patches and emit patched source plus unified diff;
- `run_user_deck`: execute a user-provided or patched DEVSIM Python deck directly and capture stdout/stderr/state;
- `plan_mutation_refinement`: read baseline-vs-mutation curve diagnostics and generate the next finer request/deck patch;
- `plan_guidance_patch`: turn `curve_guidance.next_patch_hint` into an executable request/deck patch even before a full mutation-effect state exists;
- `plan_experiment_design`: rank next experiments from signoff gaps, benchmark warnings, curve diagnostics, golden/measured availability, and deck mutations;
- `generate_report`: create a sweep/optimization report, or fall back to an engineering conclusion for single-run states;
- `generate_dashboard`: create a dashboard for a sweep, optimization, or autonomous timeline;
- `stop_success`: finish when enough evidence and artifacts exist;
- `ask_user`: stop for high-risk confirmation.

When an OpenAI-compatible model supports native tool/function calling, the agent exposes the dynamic runner registry as tool schemas and includes the industrial runner registry in context. If tool calling is unavailable, it falls back to the structured JSON action protocol.

For broad natural-language requests, use `agent_goal_router` first:

```bash
python3.11 -m tcad_agent.tools.agent_goal_router \
  --goal "AI 长时间自主操作 DEVSIM/Sentaurus 完成功率器件优化任务"
```

The router emits the selected template, preferred runner, autonomous request, Sentaurus external-workspace gate when needed, and the evidence plan.

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
- `--enable-live-evidence-lookup`: fetch matched public registry URLs before planning; findings are written to `public_evidence_lookup.json` and merged into `checkpoint.public_evidence_dossier`;
- `--allow-live-evidence-gaps`: continue only after an explicit operator decision when live lookup could not verify public evidence; the checkpoint records the override;
- `--allow-user-confirmation-actions`: allow high-risk actions that would otherwise pause;
- `--source-state-path`: resume from an existing TCAD state;
- `--source-deck-path` and `--deck-patches-json`: parse and patch a user deck before running tools;
- `--allow-unverified-deck-patch-execution`: execute even when a semantic deck patch only produced an unverified fallback append;
- `--objectives-json` and `--constraints-json`: add objective/Pareto gates to the loop;
- `--max-mutation-refinements`: limit automatic curve-guided follow-up patches;
- `--no-auto-mutation-refinement`: write the refinement work package without executing it;
- `--enable-experiment-design`: after benchmark, generate ranked convergence/golden/repair/mutation candidates and execute the highest-value candidate;
- `--max-experiment-design-rounds`: cap automatic experiment-design rounds;
- `--no-auto-experiment-design`: write the experiment-design work package without executing the selected candidate;
- `--cancel-file` and `--heartbeat-path`: cooperate with external long-run controls;
- `--require-capability-audit`: record executable/fidelity/signoff coverage before running, including a `runner_promotion_plan` for industrial or surrogate routes;
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
- semantic deck patch results record verified and unverified patches; fallback appends are warnings, not proof that the user deck uses the value;
- curve-guided mutation refinement respects the same confirmation gate for geometry/process/model changes;
- curve-guided patch execution records `guidance_patch_id`, deck patch history, overlay, and `mutation_effect_analysis` on the patched result;
- queue cancel writes an agent cancel token; the agent checks it at step boundaries and writes heartbeat state;
- DEVSIM subprocess helpers also poll the cancel token and terminate the child process when it appears;
- queued confirmation pauses can be approved or rejected through the web API;
- deterministic fallback remains available unless disabled;
- compact/planned evidence is still blocked by physical benchmark and signoff evidence gates.
- industrial runner registry entries distinguish real DEVSIM runners from physics surrogates and list remaining signoff gaps.

## Agent Experiment Design

`plan_experiment_design` is the stronger agent loop for signoff-oriented work. It does not hard-code one repair rule. It builds a ranked candidate set:

- `tool_convergence` when mesh/model/bias convergence is missing;
- `golden_curve_comparison` when a measured or golden curve path is available;
- `power_mosfet_signoff` when a completed Power MOSFET 2D field-plate run still has mesh/golden/process evidence gaps;
- `plan_mutation_refinement` when baseline-vs-mutation overlay/Pareto evidence says a direction helped and should be refined;
- `run_repair_executor` when quality or benchmark checks are failed/suspicious;
- mutation probes when the state exposes `tcad_deck_mutations`.

The selected candidate is stored in `checkpoint.pending_agent_experiment_candidate`, the full candidate set is stored in `checkpoint.agent_experiment_candidates`, and the JSON work package is written under `experiment_design/` in the agent directory.

For Sentaurus states, the same experiment-design budget first goes through `plan_sentaurus_patch`. That planner reads the latest Sentaurus state/project copy, parses deck IR, maps the natural-language goal to verified semantic patch candidates, writes `sentaurus_patch_plans/sentaurus_patch_plan_*.json`, and stores:

- `checkpoint.sentaurus_patch_candidates`;
- `checkpoint.pending_sentaurus_patch_candidate` when a safe verified candidate is selected;
- `checkpoint.blocked_sentaurus_patch_candidates` when only confirmation-gated candidates exist.

Each autonomous run also writes `checkpoint.public_evidence_dossier` before planning. The Sentaurus planner copies that gate into the patch plan, including matched public sources, convergence playbooks, model/metric expectations, and the rule that new simulator-specific operations require live lookup or local deck evidence before execution.

If automatic experiment execution is enabled, a selected low/medium-risk candidate becomes the next `sentaurus_run` request with its patches attached. After that run, `sentaurus_mutation_effect_analyzer` compares the baseline and patched states and writes `sentaurus_mutation_effect_analysis` into the patched state plus `checkpoint.latest_sentaurus_mutation_effect_analysis`. The analysis includes an engineer-style curve review for leakage-window, BV-bracket, knee, and field-peak movement, plus a machine-readable Pareto decision. The same state also receives `sentaurus_lineage_archive.json`, which compactly records the multi-run patch trail, key metrics, Pareto front, and best entry.

The next decision consumes that analysis:

- `continue_refine` triggers `plan_sentaurus_refinement` when experiment budget remains; the refiner takes a smaller verified follow-up step from the prior patch value instead of restarting from generic rules;
- `blocked_for_pareto_review` triggers configured objective/constraint evaluation or pauses for review;
- `switch_target` and `reject_candidate` trigger `plan_sentaurus_refinement` to ask the planner for a different verified target and filter out the repeated patch direction.

Refinement work packages are written under `sentaurus_patch_refinements/` and selected candidates are stored in `checkpoint.pending_sentaurus_patch_candidate`, so the normal Sentaurus execution path and confirmation gates still apply. When `use_llm=true`, the model can select among verified refinement candidates and record its engineering rationale, but it cannot create new deck patches or bypass the verification/risk gate.

High-risk geometry/process/model changes still pause for confirmation.

## Hypothesis Tree

The autonomous runtime now keeps a compact research trail in `checkpoint.agent_hypothesis_tree`. Each executed or planned step appends one node with:

- the physics/numerics hypothesis being tested;
- the expected observation and stop condition when an LLM provides them;
- the action kind, tool name, evidence keys, result state, and verdict;
- fallback alternatives when the hypothesis fails or remains suspicious.

This lets a resumed multi-hour run continue from an explicit hypothesis history instead of treating each tool call as an isolated rule. The same summary is exposed to the web cockpit and included in compact checkpoint output.
