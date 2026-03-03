"""Example: LangChain chain with ADLI — no external tracing tools.

Requires: OPENAI_API_KEY, ADLI_TOKEN
"""
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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

result = chain.invoke("Translate to French: Hello, how are you?")
print(result.content)
