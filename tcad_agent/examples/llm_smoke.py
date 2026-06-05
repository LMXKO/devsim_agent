from __future__ import annotations

import json

from tcad_agent.llm import LLMClient, LLMConfig


def main() -> None:
    config = LLMConfig.from_env()
    client = LLMClient(config)
    answer = client.chat(
        system="You are a concise TCAD assistant.",
        user="用一句话说明 TCAD agent 第一阶段最应该自动化什么。",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "base_url": config.base_url,
                "model": config.model,
                "answer": answer,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

