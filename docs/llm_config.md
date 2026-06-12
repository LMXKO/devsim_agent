# LLM Configuration

ActSoft TCAD Agent can use an OpenAI-compatible chat-completions endpoint for natural-language task decomposition, failure diagnosis, replanning, and engineering conclusion generation.

The public repository is intentionally unconfigured by default. It should not contain a real model URL, model name, API key, or private gateway address.

## Web Settings

Start the web workbench and open the settings dialog:

```bash
python3.11 -m uvicorn tcad_agent.asgi_web:app --host 127.0.0.1 --port 8766 --no-access-log
```

In the page, set:

- URL: the OpenAI-compatible base URL ending at `/v1`, for example `http://localhost:8000/v1`;
- Model: the chat model name exposed by that endpoint;
- API Key: optional. Leave it blank when the endpoint does not require authentication.

Saved settings are written to `runs/llm_settings.json`. The `runs/` directory is ignored by git and should stay local.

## Environment Variables

You can also configure the model from the shell:

```bash
export ACTSOFT_LLM_BASE_URL="http://localhost:8000/v1"
export ACTSOFT_LLM_MODEL="your-chat-model"
export ACTSOFT_LLM_API_KEY=""
export ACTSOFT_LLM_TIMEOUT_SECONDS="60"
```

The URL should point to the OpenAI-compatible API root, not to `/chat/completions`. The client appends the chat-completions path through the OpenAI SDK.

## Health Check

Use the settings dialog's LLM check action after saving configuration. It calls the same health path used by the web workbench:

```http
POST /api/llm/check
```

The check reports `unconfigured`, `passed`, or `failed` with the configured base URL, model, latency, and failure reason when available.

## Mission Planning

Use the configured model to generate the mission goal-decomposition DAG before execution:

```bash
python3.11 -m tcad_agent.tools.mission_agent \
  --mission-id mission_llm \
  --goal "帮我完成一个 MOSFET Id-Vg 任务，收敛失败时自动调步长，最后给工程结论" \
  --use-llm \
  --execute
```

If the model call fails or returns invalid JSON, the mission agent falls back to the deterministic decomposer by default. Add `--no-llm-fallback` when you want model failures to stop the mission.

## Security Notes

- Do not commit real API keys, private endpoint URLs, local gateway IPs, or generated `runs/` artifacts.
- Keep `.env` files local. Use `.env.example` only as a placeholder template.
- If you publish an existing git history, scan the full history before pushing, not just the current working tree.
