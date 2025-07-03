from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions=300)

documents = [
    'Rohit sharma is called Hit-man',
    'Sachin tendulkar is got of cricket',
    'Virat Kohli is one of the goat',
    'Jasprit Bumrah is a goat bowler',
    'Karun Nair is good'
]

query = 'tell me about virat kohli'

doc_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]
# 2d vectors are passed in cosine_similiarity , therefore query_embeddings was converted to 2d

index , score = (sorted(list(enumerate(scores)), key = lambda x:x[1])[-1])

print("query : ", query)
print(documents[index])
print('similiarity score is : ', score)