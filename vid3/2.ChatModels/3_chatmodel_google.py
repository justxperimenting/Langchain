from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro')

result = model.invoke('whatis the capital of india')

print(result.content)

# paid google api key