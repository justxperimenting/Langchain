from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


# creating embedding of a mutiple query

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bemgal",
    "Paris is the capital of France"
]


result = embedding.embed_documents(documents)

print(str(result))

