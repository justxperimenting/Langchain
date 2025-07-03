from langchain_openai import OpenAI
from dotenv import load_dotenv # used to load the variable present in .env

load_dotenv()

llm = OpenAI(model = 'gpt-3.5-turbo-instruct')

result = llm.invoke("What is the capital of india")

# the response from the chatgpt model is stored in result
print(result)


## this whole LLM model is outdated 