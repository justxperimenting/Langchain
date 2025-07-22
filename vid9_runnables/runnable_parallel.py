from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.schema.runnable import RunnableParallel,RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a very small tweet about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a very small LinkedIn post about {topic}',
    input_variables = ['topic']
)

llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2',
    task= 'text-generation'
)
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()
parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1,model,parser),
        'linkedin': RunnableSequence(prompt2,model,parser)
    }
)

result = parallel_chain.invoke({'topic':'AI'})
print(result['tweet'])