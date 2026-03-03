"""Example: LangChain chain with dual logging to ADLI + LangSmith.

LangSmith traces automatically via env vars — no code changes needed.
ADLI callback handler captures the trace through LangChain callbacks.
Both work side by side.

Requires: OPENAI_API_KEY, ADLI_TOKEN, LANGCHAIN_API_KEY
"""
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from adli_sdk import ADLI

# LangSmith — just set env vars, it traces automatically
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]

adli = ADLI(
    token=os.environ["ADLI_TOKEN"],
    project_id=int(os.environ.get("ADLI_PROJECT_ID", "1")),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful translator."),
    ("human", "{input}"),
])
model = ChatOpenAI(model="gpt-4o")
chain = prompt | model

chain = adli.wrap(chain, agent_name="translator")

result = chain.invoke("Translate to French: Hello, how are you?")
print(result.content)
# Traces sent to BOTH LangSmith and ADLI
