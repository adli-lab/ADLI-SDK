"""Example: Manual inject without wrapper — full control."""
import asyncio
import os

from pydantic_ai import Agent
from adli_sdk import ADLI


Agent.instrument_all()

adli = ADLI(token=os.environ["ADLI_TOKEN"], project_id=int(os.environ["ADLI_PROJECT_ID"]))
adli.instrument()

agent = Agent("openai:gpt-4o", system_prompt="You are a helpful assistant.")


async def main() -> None:
    # Step 1: Inject — get potentially modified message + trace_id
    inject_result = await adli.ainject("What is the weather?", agent_name="weather-agent")

    print(f"Injected: {inject_result.injected}")
    print(f"Trace ID: {inject_result.adli_trace_id}")
    print(f"Message:  {inject_result.message}")

    # Step 2: Run agent with trace_id in metadata
    result = await agent.run(
        inject_result.message,
        metadata={"adli_trace_id": inject_result.adli_trace_id},
    )

    # SpanProcessor auto-captures the trace and sends to /learn
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
