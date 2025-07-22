from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2',
    task= 'text-generation'
)

prompt1 = PromptTemplate(
    template = 'Write a small joke on {topic}',
    input_variables=['topic']
)

model = ChatHuggingFace(
    llm = llm
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = 'Explain this joke to me : \n {Joke}',
    input_variables= ['Joke']
)

chain = RunnableSequence(prompt1,model,parser)

chain.invoke({'topic':'Car'})