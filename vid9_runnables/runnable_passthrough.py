from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a small joke about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Explain the joke : {joke}',
    input_variables = ['joke']
)

llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2',
    task= 'text-generation'
)
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

joke_generator_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'explanation': RunnableSequence(prompt2,model,parser)
    }
)

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

result = final_chain.invoke({'topic':'John Cena'})
print(result)