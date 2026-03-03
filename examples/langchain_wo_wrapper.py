"""Example: LangChain manual mode — full control over inject and callbacks.

Use this approach when you need to:
- Add your own callbacks (LangFuse, LangSmith) alongside ADLI
- Control inject timing or message handling yourself
"""
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from adli_sdk import ADLI

adli = ADLI(
    token=os.environ["ADLI_TOKEN"],
    project_id=int(os.environ.get("ADLI_PROJECT_ID", "1")),
    base_url=os.environ.get("ADLI_BASE_URL", "https://api.adli.dev"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful translator."),
    ("human", "{input}"),
])
model = ChatOpenAI(model="gpt-4o")
chain = prompt | model

# Step 1: Inject
inject_result = adli.inject("Translate to French: Hello!", agent_name="translator")
print(f"Injected: {inject_result.injected}")
print(f"Trace ID: {inject_result.adli_trace_id}")

# Step 2: Create ADLI callback handler
adli_handler = adli.langchain_callback(
    agent_name="translator",
    adli_trace_id=inject_result.adli_trace_id,
    user_message=inject_result.message,
)

# Step 3: Invoke with ADLI handler (+ any other callbacks you need)
result = chain.invoke(
    {"input": inject_result.message},
    config={"callbacks": [adli_handler]},
)
print(result.content)
