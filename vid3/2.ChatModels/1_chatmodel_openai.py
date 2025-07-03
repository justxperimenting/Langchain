from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = 'gpt-4', temperature = 1.1, max_completion_tokens = 10)
# temperature parameter controls the randomness of the language model in respect to how deterministic and creative the responses would be

# max_completion_token tells the max tokens in the response


result = model.invoke('What is the capital of India')

print(result)
# result contains alots of information 

# so, to access the response only
print(result.content)

## since i don't have credits for api key in my openai account , therefore the result won't print