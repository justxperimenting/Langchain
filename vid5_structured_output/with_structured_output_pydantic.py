from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional,Literal
from pydantic import BaseModel,Field

load_dotenv()

# No idea why my hugging face model don't work
llm = HuggingFaceEndpoint(
    repo_id = 'openchat/openchat-3.5-0106',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

# schema
class Review(BaseModel):
    
    key_themes : list[str] = Field(description= "Write down all the key themes discussed in the review in a list") 
    
    summary : str = Field(description= "A brief summary of the review")
    
    sentiment : Literal['pos','neg'] = Field(description =  "Return sentiment of the review either positive, negative or neutral")
    
    pros: Optional[list[str]] = Field(default = None, description =  "Write down all the pros inside a list" )
    
    cons: Optional[list[str]] = Field(default = None, description =  "Write down all the cons inside a list" )

 
structured_model = model.with_structured_output(Review)


result = structured_model.invoke(""" The hardware is great, but the software feels bloated . There are too many pre-installed apps that i cant't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this""")

print(result)
print(result['summary'])
print(result['sentiment'])
