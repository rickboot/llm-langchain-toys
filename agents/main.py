from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder
)
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dotenv import load_dotenv

from tools.sql import run_query_tool

load_dotenv()

chat = ChatOpenAI()

tools = [run_query_tool]

prompt = ChatPromptTemplate(
  messages=[
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
  ]
)

agent = OpenAIFunctionsAgent(
  llm=chat,
  prompt=prompt,
  tools=tools
)

agent_executor = AgentExecutor(
  agent=agent,
  verbose=True,
  tools=tools
)

# agent_executor('how many users are in the database?')
agent_executor('what is the name of the first user?')