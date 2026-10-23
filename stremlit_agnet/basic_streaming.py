from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_groq import ChatGroq
import streamlit as st

# ----------------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="Groq Streaming Chat", page_icon="⚡")
st.title("⚡ Streaming Chat with Groq LLM")

"""
This example demonstrates how to stream responses from Groq’s ultra-fast models  
directly into a Streamlit chat interface using **LangChain**.
"""

# ----------------------------------------------------------
# Custom Callback Handler for Streaming Tokens
# ----------------------------------------------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# ----------------------------------------------------------
# Sidebar: API Key Input
# ----------------------------------------------------------
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you today?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# ----------------------------------------------------------
# Chat Input and Model Response
# ----------------------------------------------------------
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not groq_api_key:
        st.info("Please add your Groq API key to continue.")
        st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        # 🧠 Use Groq's Llama 3.3 70B model
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            streaming=True,
            callbacks=[stream_handler],
        )

        # Get response from Groq model
        response = llm.invoke(st.session_state.messages)

        # 🧹 Clean output to remove echoed prompt
        ai_output = response.content.strip()
        if ai_output.lower().startswith(prompt.lower()):
            ai_output = ai_output[len(prompt):].strip(" :.-\n")

        st.session_state.messages.append(ChatMessage(role="assistant", content=ai_output))
        st.write(ai_output)
