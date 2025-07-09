from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional,Literal

load_dotenv()

# No idea why my hugging face model don't work
llm = HuggingFaceEndpoint(
    repo_id = 'openchat/openchat-3.5-0106',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

# schema
json_schema = {
    "title" : "Review",
    "type" : "object",
    "properties" : {
        "key_themes" : {
            "type" : "array",
            "items": {
                "type": "string"
            },
            "description" : "Write down all key themes discussed in the review in a list"
        },
        "summary" : {
            "type" : "string",
            "description" : "a brief summary of the review"
        },
        "sentiment" : {
            "type" : "string",
            "enum" : ["pos","neg"], # we don't use literal in json
            "description" : "return sentiment of the review either positive, negative or neutral"
        },
        "pros":{
            "type" : ["array", "null"],
            "items" : {
                "type" : "string"
            },
            "description" : "Write down all the pros inside in a list"
        },
        "cons":{
            "type" : ["array", "null"],
            "items" : {
                "type" : "string"
            },
            "description" : "Write down all the cons inside in a list"
        },
        "name" : {
            "type": ["string", "null"],
            "description": "write the name of the reviewer"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke(""" The hardware is great, but the software feels bloated . There are too many pre-installed apps that i cant't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this""")

print(result)
print(result['summary'])
print(result['sentiment'])
