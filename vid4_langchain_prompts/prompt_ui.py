from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "katanemo/Arch-Router-1.5B", 
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

import streamlit as st

st.header('Research Tool')

# user_input = st.text_input('Enter your prompt') # static input


# dynamic input
paper_input = st.selectbox( 'Select the Research Paper Name', ["Select...", "Attention Is All You Need", "BERT : Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# template
template = load_prompt('template.json')

# fill the placeholders
prompt = template.invoke(
    {
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    }
)

if st.button('Summarize'):
    
    if paper_input == "Select...":
        st.warning("Please select a research paper")
    else:
        result = model.invoke(prompt)
        st.write(result.content)
        
        