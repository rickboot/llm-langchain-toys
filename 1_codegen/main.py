from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from dotenv import load_dotenv
import argparse

load_dotenv()
llm = OpenAI()

parser = argparse.ArgumentParser(description='Generate code with tests using OpenAI LLM')
parser.add_argument("--task", default="print hello world")
parser.add_argument("--language", default="python")
args = parser.parse_args()

code_prompt = PromptTemplate(
  input_variables=['language', 'task'],
  template='Write a short {language} function that will {task}'
)

test_prompt = PromptTemplate(
  input_variables=['language', 'test'],
  template='Write a unit test for this {code}'
)

code_chain = LLMChain(
  llm=llm, 
  prompt=code_prompt,
  output_key='code'
)

test_chain = LLMChain(
  llm=llm,
  prompt=test_prompt,
  output_key='test'
)

chain = SequentialChain(
  chains=[code_chain, test_chain],
  input_variables=['language', 'task'],
  output_variables=['code', 'test']
)

result = chain({
  'language': args.language,
  'task': args.task
})

print('\n\n========== Generated code ==========')
print(result['code'])
print('\n\n========== Generated test ==========')
print(result['test'])