# ADLI-SDK

Official Python SDK for [ADLI](https://adli.dev) — runtime strategy learning for AI agents.

ADLI makes your agents smarter with every run. The SDK does two things:

1. **Inject** — before each agent run, checks if there's a relevant strategy
   and modifies the user message.
2. **Learn** — after each run, captures the full conversation trace and sends
   it to ADLI for strategy evolution.

## Installation

```bash
pip install adli-sdk
```

## Quick Start

ADLI works with your agent framework directly — **no external tracing tools
required**. No Logfire, no LangFuse, no LangSmith — just your framework and
ADLI.

### PydanticAI

```python
from pydantic_ai import Agent
from adli_sdk import ADLI

Agent.instrument_all()

adli = ADLI(token="adli-xxx", project_id=1)
adli.instrument()

agent = Agent("openai:gpt-4o", system_prompt="You are a SQL analyst.")
agent = adli.wrap(agent, agent_name="sql-agent")

result = await agent.run("Get all customers with outstanding debt")
# trace captured and sent to ADLI automatically
```

### LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from adli_sdk import ADLI

adli = ADLI(token="adli-xxx", project_id=1)

chain = ChatPromptTemplate.from_template("Translate to French: {input}") | ChatOpenAI(model="gpt-4o")
chain = adli.wrap(chain, agent_name="translator")

result = chain.invoke("Hello, how are you?")
# trace captured and sent to ADLI automatically
```

### LangGraph

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from adli_sdk import ADLI

adli = ADLI(token="adli-xxx", project_id=1)

agent = create_react_agent(model=ChatOpenAI(model="gpt-4o"), tools=[...])
agent = adli.wrap(agent, agent_name="research-assistant")

result = agent.invoke({"messages": [("human", "What's the weather?")]})
# trace captured and sent to ADLI automatically
```

That's it. Three lines added to your existing code: `ADLI(...)`,
`.instrument()` (PydanticAI only), `.wrap(...)`.

---

## Already using an observability tool?

ADLI does not replace or interfere with your existing tracing setup.
Both work side by side — you keep your dashboards, ADLI learns from the
same traces.

### PydanticAI + Logfire

```python
import logfire
from pydantic_ai import Agent
from adli_sdk import ADLI

logfire.configure()              # Logfire works as before
Agent.instrument_all()

adli = ADLI(token="adli-xxx", project_id=1)
adli.instrument()

agent = adli.wrap(Agent("openai:gpt-4o"), agent_name="sql-agent")
result = await agent.run("query")
# traces go to BOTH Logfire and ADLI
```

Don't want traces in Logfire? Just use `logfire.configure(send_to_logfire=False)` or skip Logfire entirely — ADLI works either way.

### LangChain + LangFuse

```python
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from adli_sdk import ADLI

adli = ADLI(token="adli-xxx", project_id=1)
chain = adli.wrap(prompt | model, agent_name="translator")

result = chain.invoke("Hello!", config={"callbacks": [LangfuseCallbackHandler()]})
# traces go to BOTH LangFuse and ADLI
```

Don't want LangFuse? Just don't add `LangfuseCallbackHandler()` — ADLI works either way.

### LangChain + LangSmith

```python
import os
from adli_sdk import ADLI

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls-xxx"

adli = ADLI(token="adli-xxx", project_id=1)
chain = adli.wrap(prompt | model, agent_name="translator")

result = chain.invoke("Hello!")
# traces go to BOTH LangSmith and ADLI
```

Don't want LangSmith? Just don't set the env vars — ADLI works either way.

### Summary

| Observability tool | How to enable alongside ADLI | How to use ADLI alone |
|-------------------|------------------------------|----------------------|
| **Logfire** | `logfire.configure()` | `logfire.configure(send_to_logfire=False)` or skip Logfire |
| **LangFuse** | Add `LangfuseCallbackHandler()` to callbacks | Don't add it |
| **LangSmith** | Set `LANGCHAIN_TRACING_V2` + `LANGCHAIN_API_KEY` | Don't set them |

ADLI never depends on these tools and never interferes with them.

---

## Manual Mode

When you need full control over inject timing and callback setup.

### PydanticAI

```python
from pydantic_ai import Agent
from adli_sdk import ADLI

Agent.instrument_all()

adli = ADLI(token="adli-xxx", project_id=1)
adli.instrument()

agent = Agent("openai:gpt-4o", system_prompt="You are helpful.")

inj = await adli.ainject("user query", agent_name="sql-agent")
result = await agent.run(
    inj.message,
    metadata={"adli_trace_id": inj.adli_trace_id},
)
```

### LangChain / LangGraph

```python
from adli_sdk import ADLI

adli = ADLI(token="adli-xxx", project_id=1)

inj = adli.inject("user query", agent_name="my-chain")
handler = adli.langchain_callback(
    agent_name="my-chain",
    adli_trace_id=inj.adli_trace_id,
    user_message=inj.message,
)
result = chain.invoke(
    {"input": inj.message},
    config={"callbacks": [handler]},
)
```

---

## How It Works

ADLI SDK uses each framework's **native** observability mechanism:

| Framework | Mechanism | ADLI component | External deps |
|-----------|-----------|----------------|---------------|
| PydanticAI | OTel spans via `instrument_all()` | `ADLISpanProcessor` | `opentelemetry-sdk` (in deps) |
| LangChain / LangGraph | Callbacks | `ADLICallbackHandler` | `langchain-core` (user already has it) |

No Logfire, LangFuse, or LangSmith packages are required.

The `wrap()` method creates a transparent `__getattr__` proxy. All methods and
attributes are delegated to the original object. Only the entry points
(`run` / `invoke`) are intercepted to call `/inject` and attach the
appropriate trace collection mechanism.

### What gets captured

| Data | PydanticAI | LangChain / LangGraph |
|------|------------|----------------------|
| System prompt | via `pydantic_ai.all_messages` | via `on_chat_model_start` |
| User message | via `pydantic_ai.all_messages` | from wrapper inject |
| AI responses | via `pydantic_ai.all_messages` | via `on_llm_end` |
| Tool calls | via `pydantic_ai.all_messages` | via `on_llm_end` (tool_calls) |
| Tool results | via `pydantic_ai.all_messages` | via `on_tool_end` |
| Token usage | via `gen_ai.usage.*` span attrs | via `usage_metadata` / `llm_output` |
| Errors | via span status | via `on_chain_error` |

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full design document.

## License

Apache 2.0 — see [LICENSE](LICENSE).
