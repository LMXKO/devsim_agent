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

## Semantic Patches

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
python3.11 -m tcad_agent.tools.autonomous_devsim_agent \
  --goal "用 Sentaurus 跑这个 LDMOS 项目，降低漏电，同时不要牺牲 BV/Ron，必要时提出下一轮 deck patch" \
  --sentaurus-project-path /Users/me/tcad_projects/ldmos_case \
  --sentaurus-profile-path ~/.actsoft/sentaurus_profile.json \
  --sentaurus-request-json '{"flow":["sdevice"],"deck_files":["device.cmd"],"timeout_seconds":7200}' \
  --enable-experiment-design \
  --execute
```

The first autonomous action becomes `sentaurus_run`. The resulting state then flows through physical benchmark, objective/Pareto evaluation, report generation, and agent experiment design.

## Test Boundary

Unit tests use fake external commands to validate process control, patching, logs, CSV ingestion, benchmark integration, and autonomous routing. Fake commands do not represent real Sentaurus physics. New simulated scenarios should be grounded in official/public documentation or user-provided real project evidence before being encoded.

