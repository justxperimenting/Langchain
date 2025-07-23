from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

loader = TextLoader('cricket.txt',encoding = 'utf-8')

llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2',
    task= 'text-generation'
)
model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template= 'Write the summary for the following poem \n {Poem}',
    input_variables=['Poem']
)

parser = StrOutputParser()
docs = loader.load()

chain = prompt | model | parser
result = chain.invoke({'Poem':docs[0].page_content})

print(result)