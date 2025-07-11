from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

from langchain.output_parsers import StructuredOutputParser, ResponseSchema


from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name = 'fact_1', description = 'Fact 1 about the topic'),    
    ResponseSchema(name = 'fact_2', description = 'Fact 2 about the topic'),    
    ResponseSchema(name = 'fact_3', description = 'Fact 3 about the topic'),    
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give 3 facts about the {topic} \n {format_instruction}',
    input_variables = ['topic','format_instruction'],
)

prompt = template.format(
    topic= 'black hole',format_instruction = parser.get_format_instructions()
    )

result = model.invoke(prompt)

final_result = parser.parse(result.content)



print( final_result)
