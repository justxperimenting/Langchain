from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# done to make sure that the llm is downloaded in d drive
import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id= 'katanemo/Arch-Router-1.5B',
    task = 'text-generation',
    pipeline_kwargs= dict(
        temperature = 0.5,
        max_new_tokens = 10
    )
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("what is the capital of india?")

print(result.content)