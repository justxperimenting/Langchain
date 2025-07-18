from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2',
    task= 'text-generation'
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    
    sentiment : Literal['positive','negative'] = Field(description = 'Give the sentiment of the feedback')
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of following feedback text into positive or negative \n {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables= {'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = "Write an appropriate response to this positive feedback \n {feedback}",
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = "Write an appropriate response to this negative feedback \n {feedback}",
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positve', prompt2 | model | parser),
    (lambda x : x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x : "could not find sentiment") # since lambda function is not a chain so we converted this function into a chain using RunnableLambda
)

chain = classifier_chain | branch_chain

chain.get_graph().print_ascii()