# ADLI-SDK

Official Python SDK for [ADLI](https://adli.dev) — runtime strategy learning for AI agents.

ADLI makes your agents smarter with every run. The SDK does two things:

1. **Inject** — before each run, checks if there's a relevant strategy and modifies the user message.
2. **Learn** — after each run, captures the full conversation trace and sends it to ADLI for strategy evolution.

## Installation

```bash
pip install adli-sdk
```

## Quick Start

Three lines added to your existing code: `ADLI(...)`, `.instrument()` (PydanticAI only), `.wrap(...)`.

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
```

### LangChain / LangGraph

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from adli_sdk import ADLI

adli = ADLI(token="adli-xxx", project_id=1)

chain = ChatPromptTemplate.from_template("Translate to French: {input}") | ChatOpenAI(model="gpt-4o")
chain = adli.wrap(chain, agent_name="translator")

result = chain.invoke("Hello, how are you?")
```

LangGraph compiled graphs work identically — `create_react_agent(...)` returns a `Runnable`, so `adli.wrap(agent, ...)` works as-is.

### CrewAI

```python
from crewai import Crew, Agent, Task, Process
from adli_sdk import ADLI

adli = ADLI(token="adli-xxx", project_id=1)

crew = Crew(agents=[...], tasks=[...], process=Process.sequential)
crew = adli.wrap(crew, agent_name="research-crew")

result = crew.kickoff(inputs={"topic": "AI trends in 2025"})
```

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from adli_sdk import ADLI

adli = ADLI(token="adli-xxx", project_id=1)

index = VectorStoreIndex.from_documents(SimpleDirectoryReader("data").load_data())
query_engine = index.as_query_engine()
query_engine = adli.wrap(query_engine, agent_name="rag-agent")

response = query_engine.query("What are the key findings?")
```

ChatEngine works the same way — `adli.wrap(index.as_chat_engine(), agent_name="...")`.

### OpenAI Agents SDK

```python
from agents import Agent, Runner, RunConfig
from adli_sdk import ADLI

adli = ADLI(token="adli-xxx", project_id=1)
adli.instrument_openai_agents()  # register once at startup

agent = Agent(name="assistant", instructions="You are a helpful assistant.", tools=[...])

# inject manually, then run
inj = adli.inject("user query", agent_name="assistant")
result = await Runner.run(
    agent,
    inj.message,
    run_config=RunConfig(metadata={
        "adli_trace_id": inj.adli_trace_id,
        "adli_agent_name": "assistant",
        "adli_user_message": inj.message,
    }),
)
```

---

## Already using an observability tool?

ADLI does not replace or interfere with your existing tracing setup. Both work side by side.

- **Logfire**: call `logfire.configure()` before `Agent.instrument_all()` — traces go to both.
- **LangFuse**: add `LangfuseCallbackHandler()` to your chain's `config={"callbacks": [...]}` — runs alongside ADLI's handler.
- **LangSmith**: set `LANGCHAIN_TRACING_V2=true` as usual — ADLI's callback handler works independently.

ADLI never depends on these tools and never interferes with them.

---

## Manual Mode

For full control over inject timing and callback setup.

### PydanticAI

```python
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
adli = ADLI(token="adli-xxx", project_id=1)

inj = adli.inject("user query", agent_name="my-chain")
handler = adli.langchain_callback(
    agent_name="my-chain",
    adli_trace_id=inj.adli_trace_id,
    user_message=inj.message,
)
result = chain.invoke({"input": inj.message}, config={"callbacks": [handler]})
```

---

## How It Works

| Framework | Mechanism | ADLI component |
|-----------|-----------|----------------|
| PydanticAI | OTel spans via `instrument_all()` | `ADLISpanProcessor` |
| LangChain / LangGraph | LangChain callbacks | `ADLICallbackHandler` |
| CrewAI | LLM-level callbacks | `ADLICallbackHandler` (reused) |
| LlamaIndex | `CallbackManager` events | `ADLILlamaIndexHandler` |
| OpenAI Agents SDK | `TracingProcessor` | `ADLIAgentsProcessor` |

The `wrap()` method creates a transparent `__getattr__` proxy. All attributes and methods are delegated to the original object. Only entry points (`run` / `invoke` / `kickoff` / `query` etc.) are intercepted to call `/inject` and attach the trace collection mechanism.

## License

Apache 2.0 — see [LICENSE](LICENSE).
