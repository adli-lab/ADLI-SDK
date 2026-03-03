"""Example: PydanticAI agent with ADLI — no external tracing tools.

Requires: OPENAI_API_KEY, ADLI_TOKEN
"""
import asyncio
import os

from pydantic_ai import Agent

from adli_sdk import ADLI

Agent.instrument_all()

adli = ADLI(
    token=os.environ["ADLI_TOKEN"],
    project_id=int(os.environ.get("ADLI_PROJECT_ID", "1")),
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
    # Trace sent to ADLI automatically


if __name__ == "__main__":
    asyncio.run(main())
