from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'openchat/openchat-3.5-0106',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

# schema
class Review(TypedDict):
    
    key_themes : Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    
    summary : Annotated[str, "A brief summary of the review"]
    
    sentiment : Annotated[str, "Return sentiment of the review either positive, negative or neutral"]
 
structured_model = model.with_structured_output(Review)


result = structured_model.invoke(""" The hardware is great, but the software feels bloated . There are too many pre-installed apps that i cant't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this""")

print(result)
