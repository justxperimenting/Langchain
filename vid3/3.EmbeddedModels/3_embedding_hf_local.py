from langchain_huggingface import HuggingFaceEmbeddings

# files installed will be in d drive
import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the capital of India"

vector = embedding.embed_query(text)

print(str(vector))


# code nhi chal rha , dimag kharab ho gya h :(