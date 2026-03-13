# ADLI-SDK

Official Python SDK for [ADLI](https://adli.dev) — runtime strategy learning for AI agents.

ADLI makes your agents smarter with every run:

1. **Inject** — before each run, checks if there's a relevant strategy and modifies the user message.
2. **Learn** — after each run, captures the full conversation trace (messages, tool calls, reasoning, usage) and sends it to ADLI for strategy evolution.

## Installation

```bash
pip install adli-sdk
```

## Quick Start

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

### LangChain

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from adli_sdk import ADLI

@tool
def lookup_schema(table_name: str) -> str:
    """Look up the database schema for a table."""
    return f"{table_name}(id, name, created_at)"

adli = ADLI(token="adli-xxx", project_id=1)

model = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior SQL analyst. Use tools to check table structures."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(model, [lookup_schema], prompt)
executor = AgentExecutor(agent=agent, tools=[lookup_schema])

executor = adli.wrap(executor, agent_name="sql-agent")
result = executor.invoke({"input": "What's the schema of the customers table?"})
```

### LangGraph

Multi-node `StateGraph` with tools, conditional edges, and tool loops — works the same way. Pass `input_key="messages"` for `MessagesState`.

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from adli_sdk import ADLI

@tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Results for '{query}'..."

model = ChatOpenAI(model="gpt-4o")
tools = [search_docs]

def researcher(state: MessagesState) -> dict:
    sys = SystemMessage(content="You are a researcher. Use tools to gather info.")
    return {"messages": [model.bind_tools(tools).invoke([sys] + state["messages"])]}

def synthesizer(state: MessagesState) -> dict:
    sys = SystemMessage(content="Summarize all findings into a final answer.")
    return {"messages": [model.invoke([sys] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("researcher", researcher)
builder.add_node("tools", ToolNode(tools))
builder.add_node("synthesizer", synthesizer)
builder.add_edge(START, "researcher")
builder.add_conditional_edges("researcher", tools_condition, {"tools": "tools", "__end__": "synthesizer"})
builder.add_edge("tools", "researcher")
builder.add_edge("synthesizer", END)
graph = builder.compile()

adli = ADLI(token="adli-xxx", project_id=1)
graph = adli.wrap(graph, agent_name="research-agent", input_key="messages")

result = graph.invoke({
    "messages": [HumanMessage(content="Search for auth docs and summarize.")],
})
```

ADLI captures the full graph execution: system prompts from each node, all LLM responses, tool calls with arguments, tool returns with results — correctly interleaved and paired by `tool_call_id`.

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
adli.instrument_openai_agents()

agent = Agent(name="assistant", instructions="You are a helpful assistant.", tools=[...])

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

## Works Alongside Your Observability Stack

ADLI does not replace or interfere with your existing tracing setup:

- **Logfire**: call `logfire.configure()` before `Agent.instrument_all()` — traces go to both.
- **LangFuse**: add `LangfuseCallbackHandler()` to your chain's `config={"callbacks": [...]}` — runs alongside ADLI's handler.
- **LangSmith**: set `LANGCHAIN_TRACING_V2=true` as usual — ADLI's callback handler works independently.

---

## How It Works

| Framework | Trace mechanism | Intercepted methods |
|---|---|---|
| PydanticAI | OTel `SpanProcessor` | `run`, `run_sync`, `run_stream`, `iter` |
| LangChain / LangGraph | `BaseCallbackHandler` | `invoke`, `ainvoke`, `stream`, `astream` |
| CrewAI | LangChain callbacks (reused) | `kickoff`, `kickoff_async` |
| LlamaIndex | `CallbackManager` events | `query`, `aquery`, `chat`, `achat` |
| OpenAI Agents SDK | `TracingProcessor` | Manual inject + `Runner.run` |

`wrap()` creates a transparent `__getattr__` proxy — all attributes and methods delegate to the original object. Only entry-point methods are intercepted to call `/inject` and attach trace collection.

### What Gets Captured

- System prompts (including per-node prompts in LangGraph)
- User messages
- LLM responses (text + reasoning/thinking content)
- Tool calls (name, arguments, tool_call_id)
- Tool returns (result, matched to call by tool_call_id)
- Token usage (input, output, cache)
- Outcome (success/failure)

## License

Apache 2.0 — see [LICENSE](LICENSE).
