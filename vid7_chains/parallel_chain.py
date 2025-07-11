from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2',
    task = 'text-generation'
)

llm2 = HuggingFaceEndpoint(
    repo_id= 'KurmaAI/AQUA-7B',
    task = 'text-generation'
)

# llm3 = HuggingFaceEndpoint(
#     repo_id= 'K-intelligence/Midm-2.0-Base-Instruct',
#     task = 'text-generation'
# )



model1 = ChatHuggingFace(llm = llm1)
# model2 = ChatHuggingFace(llm = llm2)
model2 = ChatHuggingFace(llm = llm2)

prompt1 = PromptTemplate(
    template = 'Generate a short and simple notes from following text {text}',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'Generate a 5 question answers from following text \n {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the following notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables = ['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
    Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
"""

result = chain.invoke({'text' : text})

# print(result)

chain.get_graph().print_ascii()


