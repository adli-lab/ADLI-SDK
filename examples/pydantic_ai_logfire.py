"""Example: PydanticAI with dual logging to ADLI + Logfire.

Logfire receives traces as usual, ADLI learns from the same OTel spans.

Requires: OPENAI_API_KEY, ADLI_TOKEN, LOGFIRE_TOKEN
"""
import asyncio
import os

import logfire
from pydantic_ai import Agent

from adli_sdk import ADLI

# Logfire — configure as usual
logfire.configure()
Agent.instrument_all()

# ADLI — plugs into the same OTel pipeline
adli = ADLI(
    token=os.environ["ADLI_TOKEN"],
    project_id=int(os.environ.get("ADLI_PROJECT_ID", "1")),
    base_url=os.environ.get("ADLI_BASE_URL", "https://api.adli.dev"),
)
adli.instrument()

agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a senior SQL analyst. Write efficient queries.",
)
agent = adli.wrap(agent, agent_name="sql-agent")


async def main() -> None:
    result = await agent.run("Get all customers with outstanding debt over $10k")
    print(result.output)
    # Traces sent to BOTH Logfire and ADLI


if __name__ == "__main__":
    asyncio.run(main())
