from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose=True)

memory = ConversationSummaryMemory(
  memory_key="messages", 
  llm=chat,
  chat_memory=FileChatMessageHistory("messages.json"),
  return_messages=True)

prompt = ChatPromptTemplate(
  input_variables = ["content", "messages"],
  messages = [
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{content}"),
  ]
)

chain = LLMChain(
  llm=chat,
  memory=memory,
  prompt=prompt,
  verbose=True
)

while True:
  content = input(">> ")
  result = chain({"content": content})
  print(result["text"])