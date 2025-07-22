# lets make a runnable lambda to count the number of words in a joke

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2',
    task= 'text-generation'
)
model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template = "Write a very small joke about {topic}",
    input_variables= ['topic']
)

parser = StrOutputParser()

def word_count(text):
    return len(text.split())

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'word_count': RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'AI'})

final_result = """{} \n word count - {}""".format(result['joke'],result['word_count'])

print(final_result)
