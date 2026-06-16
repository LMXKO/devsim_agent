# Sentaurus Adapter

`tcad_agent.sentaurus` is a local adapter for user-owned Sentaurus installations. It is designed for the product goal where a user gives a natural-language TCAD task and the agent keeps operating over a long horizon, while commercial software and confidential process assets stay outside the repository.

## Boundary

The repository does not include:

- Synopsys binaries or installers;
- license server values or license files;
- private PDKs, calibrated model files, process decks, or measured data;
- generated Sentaurus artifacts from user projects.

The adapter only stores code and schemas. Runtime profiles, project paths, license environment, and site wrapper scripts remain local.

## Runtime Profile

Save profiles outside git, for example `~/.actsoft/sentaurus_profile.json`.

```json
{
  "profile_id": "local_sentaurus",
  "sentaurus_home": "/opt/synopsys/sentaurus",
  "commands": {
    "sdevice": "/opt/synopsys/sentaurus/bin/sdevice",
    "svisual": "/opt/synopsys/sentaurus/bin/svisual"
  },
  "allowed_project_roots": ["/Users/me/tcad_projects"],
  "run_root": "/Users/me/tcad_runs/actsoft_sentaurus",
  "env": {
    "STROOT": "/opt/synopsys/sentaurus"
  },
  "default_flow": ["sdevice"],
  "curve_globs": ["*.csv", "*_extract.csv", "*_iv.csv"],
  "artifact_globs": ["*.log", "*_des.log", "*.plt", "*_des.plt", "*.tdr", "*_des.tdr", "*.csv"]
}
```

`runtime_profile` in `sentaurus_state.json` records only a safe summary: profile id, configured command names, allowed roots, default flow, glob lists, and environment variable keys.

## State Contract

Each run writes:

- `sentaurus_state.json`: durable state consumed by autonomous agent, queue, benchmark, reports, and objective evaluation;
- `project/`: controlled copy of the input project;
- `sentaurus_patch.diff`: unified diff when patches apply;
- `sentaurus_deck_ir_*.json`: parsed command-file IR for supported deck files;
- `*_stdout.log` and `*_stderr.log`: captured command output;
- copied-project artifacts matching `artifact_globs`.

The quality report classifies:

- license checkout text as `sentaurus_license_issue`;
- Newton/continuation failures as `sentaurus_convergence_issue`;
- mesh/grid text as `sentaurus_mesh_issue`;
- fatal process text or non-zero return code as failed quality;
- missing CSV extraction as suspicious, not as a fake pass.

## Curve Extraction

The agent does not parse proprietary `.plt` or `.tdr` files directly. Configure Sentaurus Visual, Inspect, or a site wrapper to export CSV into the copied project. Preferred column names are unit-bearing:

```csv
voltage_v,current_a,electric_field_v_per_cm
0,1e-12,1e4
-10,1e-9,2e5
-20,1e-6,8e5
```

With CSV present, the state records curve points, inferred x/y/field columns, leakage interval, breakdown threshold bracket, knee/shape features, field peak, and solver provenance.

## Deck IR And Semantic Patches

`tcad_agent.sentaurus_deck` parses common Sentaurus command-file structure into a conservative IR. The parser is based on public command-file examples and recognizes:

- top-level sections and nested blocks such as `File`, `Electrode`, `Physics`, `Plot`, `Math`, `Solve`, `Goal`, `Coupled`, and `Quasistationary`;
- anonymous records such as `{ Name="drain" Voltage=0.0 }` inside `Electrode`;
- assignments such as `Iterations=20`, `Grid="@tdr@"`, and `Voltage=0.0`;
- `set NAME value` and `#define NAME value` variables.

The IR keeps line numbers, block paths, assignments, variables, and warnings. It is not a complete proprietary Sentaurus grammar and must fail loud when it cannot verify a requested edit.

Supported semantic operations:

- `sentaurus_set_variable`: update a `set` or `#define` variable;
- `sentaurus_update_assignment`: update an existing assignment inside a section/block;
- `sentaurus_upsert_assignment`: update an assignment or insert it before the target block closes;
- `sentaurus_add_model`: insert a model line into a target section when absent.

Example semantic patch:

```json
[
  {
    "file": "device.cmd",
    "operation": "sentaurus_set_variable",
    "variable": "DRIFT_DOPING",
    "value": "8e14",
    "reason": "lower drift doping as a BV/leakage experiment"
  },
  {
    "file": "device.cmd",
    "operation": "sentaurus_update_assignment",
    "section_path": ["Electrode"],
    "selector": {"Name": "drain"},
    "parameter": "Voltage",
    "value": -1200,
    "reason": "move the drain reverse-bias target"
  },
  {
    "file": "device.cmd",
    "operation": "sentaurus_upsert_assignment",
    "section_path": ["Math"],
    "parameter": "Iterations",
    "value": 80,
    "reason": "allow more Newton iterations for a hard bias ramp"
  }
]
```

Raw compatibility patch operations are still available:

Patch operations are explicit and verified against the copied project:

```json
[
  {
    "file": "device.cmd",
    "operation": "replace_text",
    "pattern": "set DRIFT_DOPING 1e15",
    "replacement": "set DRIFT_DOPING 8e14",
    "reason": "lower drift doping as a BV/leakage experiment"
  },
  {
    "file": "actsoft_params.json",
    "operation": "json_set",
    "json_path": "drift.doping_cm3",
    "value": 800000000000000.0,
    "reason": "site parameter contract"
  }
]
```

Unsupported, unmatched, or path-escaping patches are recorded as unverified. Treat unverified patches as blocked for execution unless the user explicitly approves the risk.

## Autonomous Entry Point

```bash
python3.11 -m tcad_agent.autonomous_devsim_agent \
  --goal "用 Sentaurus 跑这个 LDMOS 项目，降低漏电，同时不要牺牲 BV/Ron，必要时提出下一轮 deck patch" \
  --sentaurus-project-path /Users/me/tcad_projects/ldmos_case \
  --sentaurus-profile-path ~/.actsoft/sentaurus_profile.json \
  --sentaurus-request-json '{"flow":["sdevice"],"deck_files":["device.cmd"],"timeout_seconds":7200}' \
  --enable-experiment-design \
  --execute
```

The first autonomous action becomes `sentaurus_run`. With `--enable-experiment-design`, the next Sentaurus-specific planning step is `plan_sentaurus_patch`: it inspects the latest `sentaurus_state.json`, parses the configured deck files, generates verified semantic patch candidates, writes a JSON work package under `sentaurus_patch_plans/`, and can execute the selected low/medium-risk candidate as the next `sentaurus_run`.

After that patched run, `sentaurus_mutation_effect_analyzer` compares baseline vs patched state and writes its decision back into the patched state. If experiment budget remains, `sentaurus_patch_refiner` consumes that decision: useful directions become smaller verified follow-up patches, failed directions switch to a different verified target, and Pareto tradeoffs pause for review. Every patched Sentaurus state also writes `sentaurus_lineage_archive.json` with the multi-run patch/effect/metric trail, Pareto front, and best entry. High-risk geometry/process/model classes and tradeoff regressions remain confirmation-gated.

## Real-Ready Preflight

Before running a real user-owned Sentaurus project, gate it explicitly:

```bash
python3.11 -m tcad_agent.sentaurus_preflight \
  --project /Users/me/tcad_projects/ldmos_case \
  --profile ~/.actsoft/sentaurus_profile.json \
  --deck-file device.cmd \
  --output /tmp/sentaurus_preflight.json \
  --report /tmp/sentaurus_preflight.md
```

The preflight checks that the external profile loads, flow commands resolve, project paths stay inside allowed roots, deck files parse through the conservative IR, license-related environment hints are present, and CSV extraction globs are configured. It records only safe profile summaries and environment key names, never license values.

When the machine is not ready, use the real-project long-run gate:

```bash
python3.11 -m tcad_agent.long_run_validation \
  --suite autonomous_e2e \
  --scenario-id real_sentaurus_live_llm_project_soak
```

This produces a blocked JSON/report such as `blocked_missing_sentaurus_installation`, rather than pretending to run Sentaurus.

## Patch Planner

Use the patch planner directly when Sentaurus is unavailable or when you want a reviewable work package before running:

```bash
python3.11 -m tcad_agent.sentaurus_patch_planner \
  --goal "Ramp reverse BV to 1200V, reduce step size if convergence is difficult" \
  --project tcad_agent/examples/sentaurus_fixtures/power_diode_bv \
  --deck-file device.cmd \
  --output /tmp/sentaurus_patch_plan.json
```

The planner returns:

- candidate hypotheses, expected observations, stop conditions, and fallback alternatives;
- semantic patches using the same `sentaurus_set_variable`, `sentaurus_update_assignment`, and `sentaurus_upsert_assignment` schema as `sentaurus_run`;
- validation records and diffs from applying each patch to the current deck in memory;
- a selected candidate only when every patch verifies and the risk gate allows it.

The current vocabulary recognizes continuation/Math controls, BV `Goal` voltage, drift doping, lifetime, trap density, field plate, guard ring, oxide thickness, implant dose, junction depth, trench corner radius, and region-specific lifetime variables when those variables already exist in the deck. It does not invent proprietary process syntax; unsupported targets stay as unselected or confirmation-gated candidates until real project evidence or public/official documentation justifies a schema.

## Mutation Schema Extension

When a real deck exposes a target that is not in the static mutation vocabulary, generate a review package instead of inventing an executable patch:

```bash
python3.11 -m tcad_agent.mutation_schema_agent \
  --goal "Reduce reverse leakage by tuning surface recombination velocity" \
  --project /Users/me/tcad_projects/ldmos_case \
  --deck-file device.cmd \
  --target "surface recombination velocity" \
  --output-dir /tmp/mutation_schema_extension
```

The package contains a proposed vocabulary entry, public-evidence gate, local deck variable binding, fixture deck, and semantic patch validation records. It never edits `tcad_agent.mutation_vocabulary` directly and never runs the solver. The autonomous agent can invoke the same action after `sentaurus_patch_planner` returns no actionable candidates for the current state.

## Mutation Effect Analyzer

Use the analyzer directly to compare a baseline run against a patched run:

```bash
python3.11 -m tcad_agent.sentaurus_mutation_effect \
  --baseline /tmp/sentaurus_base/sentaurus_state.json \
  --mutation /tmp/sentaurus_patch/sentaurus_state.json \
  --goal "降低漏电，同时不要牺牲 BV/Ron/field peak" \
  --candidate-json '{"candidate_id":"device.cmd:lifetime:LIFETIME_SCALE"}' \
  --output /tmp/sentaurus_mutation_effect.json
```

The analyzer reads `quality_report.metrics`, `final_summary.artifacts.sentaurus_curve_csv`, and curve shape metadata. It reports primary metric movement, improved/regressed metrics, BV bracket movement, leakage interval, field peak value/location, overlay SVG path, an engineer-style curve review, Pareto summary, a machine-readable Pareto decision, and a decision:

- `continue_refine`: primary metric or run quality improved without blocking tradeoffs;
- `blocked_for_pareto_review`: primary direction helped but BV/Ron/field/leakage constraints regressed beyond tolerance;
- `switch_target`: primary metric did not improve;
- `reject_candidate`: patched run regressed quality/status;
- `insufficient_evidence`: comparable metrics or CSV evidence are missing.

The open mutation vocabulary is schema-backed in `tcad_agent.mutation_vocabulary`. Current Sentaurus candidate classes include `field_plate`, `drift_doping`, `lifetime`, `region_specific_lifetime`, `trap_density`, `guard_ring`, `junction_depth`, `oxide_thickness`, `implant_dose`, and `trench_corner_radius`. High-risk geometry/process/model entries remain confirmation-gated even when their semantic patch validates against the deck.

## Patch Refiner And Lineage

Use the refiner after an analyzer result has been written into a patched state:

```bash
python3.11 -m tcad_agent.sentaurus_patch_refiner \
  --state /tmp/sentaurus_patch/sentaurus_state.json \
  --goal "继续降低漏电，同时不要牺牲 BV/Ron/field peak" \
  --use-llm \
  --output /tmp/sentaurus_patch_refinement.json
```

The refiner never guesses proprietary syntax. It reuses the semantic patch schema and validates the next candidate against the current copied deck. For `continue_refine`, numeric variable/assignment edits are advanced by a half step beyond the last verified old-to-new movement. For `switch_target` or `reject_candidate`, it asks the planner for alternative verified candidates and filters out the repeated patch.

With `--use-llm`, the model can choose among already verified candidates and write a rationale, expected observation, and stop condition. It cannot introduce a new patch or bypass high-risk confirmation gates. `--no-llm-fallback` makes an invalid model selection fail closed.

Build the lineage archive directly when debugging a multi-run chain:

```bash
python3.11 -m tcad_agent.sentaurus_lineage \
  --state /tmp/sentaurus_patch/sentaurus_state.json \
  --output /tmp/sentaurus_lineage_archive.json
```

The archive follows `repair_context.baseline_state_path`, records compact metrics, candidate patches, analyzer decisions, overlays, Pareto front membership, and the current best entry.

## Replay Harness

When Sentaurus is not installed but you have exported adapter states, logs, and CSV curves from another machine, replay them without running a solver:

```bash
python3.11 -m tcad_agent.sentaurus_replay \
  --baseline /runs/sentaurus_base/sentaurus_state.json \
  --mutation /runs/sentaurus_patch/sentaurus_state.json \
  --goal "降低漏电，同时不要牺牲 BV/Ron/field peak" \
  --candidate-json '{"candidate_id":"device.cmd:lifetime:LIFETIME_SCALE"}' \
  --output-dir /tmp/sentaurus_replay
```

Replay validates the state contract, checks curve availability, runs mutation-effect analysis when baseline and mutation states are provided, and builds a lineage archive. It sets `sentaurus_replay_only=true` and `tcad_solver_invoked=false` in its own summary so replay evidence cannot be confused with a fresh Sentaurus execution.

## Test Boundary

Unit tests use fake external commands to validate process control, patching, logs, CSV ingestion, benchmark integration, and autonomous routing. Fake commands do not represent real Sentaurus physics. New simulated scenarios should be grounded in official/public documentation or user-provided real project evidence before being encoded.

## Offline Contract Harness

When a real Sentaurus installation is unavailable, use the contract harness instead of inventing solver behavior:

```bash
python3.11 -m tcad_agent.sentaurus_contract \
  --all-fixtures \
  --fixtures-root tcad_agent/examples/sentaurus_fixtures

python3.11 -m tcad_agent.sentaurus_contract \
  --project tcad_agent/examples/sentaurus_fixtures/power_diode_bv \
  --run-fake-e2e \
  --output-root /tmp/actsoft_sentaurus_contract_smoke
```

The fixture corpus lives under `tcad_agent/examples/sentaurus_fixtures/` and currently covers:

- `power_diode_bv`: `File`, `Electrode`, `Physics`, `Plot`, `Math`, `Solve`, `Quasistationary`, and `Goal`;
- `mosfet_idvg`: MOS-style electrode records, `Plot(Collected)`, gate-bias ramp, and Math upsert patches;
- `mixed_mode_transient`: `Device`, `System`, mixed-mode style voltage sources, `Math(Method=Blocked)`, and `Transient`.

Each fixture has an `actsoft_sentaurus_contract.json` manifest that declares required section paths, variables, assignments, semantic patch smoke tests, required CSV columns, and optional fake-backend interface outputs. The fake backend writes logs, a placeholder `.plt`, and a CSV with manifest-declared columns; it is marked `interface_contract_only` and exists only to validate agent IO and state lineage.

This harness is the recommended way to keep building the Sentaurus agent without Sentaurus installed: strengthen parsing, patch planning, artifact lineage, and long-horizon control now, then swap the fake backend for a real `sentaurus_profile.json` later.
