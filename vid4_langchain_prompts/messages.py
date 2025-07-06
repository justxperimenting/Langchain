from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'katanemo/Arch-Router-1.5B',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

# HumanMessage : messages send by human to ai
# Aimessage : messages send by ai back to human
# SystemMessage : system level messages that are done at the start of conversation

messages = [
    SystemMessage(content= 'Your are a helpful assistant'),
    HumanMessage(content = 'Tell me about Langchain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)

