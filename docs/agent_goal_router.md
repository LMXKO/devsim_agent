# Agent Goal Router

`tcad_agent.agent_goal_router` is the top-level natural-language entry point.

It turns broad goals such as:

```bash
python3.11 -m tcad_agent.agent_goal_router \
  --goal "AI 长时间自主操作 DEVSIM/Sentaurus 完成功率器件优化任务"
```

into:

- selected device template and preferred runner;
- autonomous agent request with capability audit and experiment design enabled;
- Sentaurus external-workspace gate when commercial TCAD is requested;
- evidence plan for baseline, benchmark, experiment design, and signoff pack;
- industrial runner coverage for the selected template.

The router does not copy commercial software, licenses, PDKs, process decks, or model files. If Sentaurus is requested without a user-owned project/profile, it returns `needs_input`.
