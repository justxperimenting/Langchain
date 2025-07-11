from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field


from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    
    name: str = Field(description = 'name of the person')
    age: int = Field(gt = 18, description='age of the person', lt = 70)
    city : str = Field(description='city in which the person lives')
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = 'Generate the name ,age ,city of a fictional {place} person \n {format_instruction}',
    input_variables = ['place'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser 
result = chain.invoke({'place':'nigerian'})

print(result)

