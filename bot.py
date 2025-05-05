import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Conversational Q&A ChatBot")
st.title("Groq Chat Bot With Memory")

model_name=st.sidebar.selectbox(
    "Select Groq Model",
    ["gemma2-9b-it","llama-3.1-8b-instant","deepseek-r1-distill-llama-70b"]
)

temperature=st.sidebar.slider(
    "Temperature",0.0,1.0,0.7
)
max_tokens=st.sidebar.slider(
    "Max Tokens",50,300,150
)

if "memory" not in st.session_state:
    st.session_state.memory=ConversationBufferMemory(
        return_messages=True
    )
if "history" not in st.session_state:
    st.session_state.history=[]

user_input= st.chat_input("You:")

if user_input:
    st.session_state.history.append(("user",user_input))

    llm=ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    conv=ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )

    ai_response=conv.predict(input=user_input)

    st.session_state.history.append(("assistant",ai_response))

    for role,text in st.session_state.history:
        if role=="user":
            st.chat_message("user").write(text)
        else:
            st.chat_message("assistant").write(text)
