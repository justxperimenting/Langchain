# lets make a runnable lambda to count the number of words in a joke

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough, RunnableBranch
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2',
    task= 'text-generation'
)
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = "Summarize the following text \n {text}",
    input_variables= ['text']
)

parser = StrOutputParser()

def word_count(text):
    return len(text.split())

report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 500, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

result = final_chain.invoke({'topic':'India vs Pakistan'})

# final_result = """{} \n word count - {}""".format(result['joke'],result['word_count'])

print(result)
