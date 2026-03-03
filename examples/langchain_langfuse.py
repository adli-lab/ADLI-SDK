"""Example: LangChain chain with dual logging to ADLI + LangFuse.

LangFuse callback runs alongside ADLI callback — both receive the trace.

Requires: OPENAI_API_KEY, ADLI_TOKEN, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
"""
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from adli_sdk import ADLI

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

# LangFuse callback coexists with ADLI callback — both receive the trace
result = chain.invoke(
    "Translate to French: Hello, how are you?",
    config={"callbacks": [LangfuseCallbackHandler()]},
)
print(result.content)
# Traces sent to BOTH LangFuse and ADLI
