from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name,age and city of a fictional person \n {format_instructions}",
    input_variables = [],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
    # it is called partial variable as it gets filled before runtime
)

chain = template | model | parser 
result = chain.invoke({})   # {} is blank dicionary , it is sent as its a condition to sent dictionary 

print( result)