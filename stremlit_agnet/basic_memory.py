import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

"""
A basic example of using **StreamlitChatMessageHistory** with **LangChain** and **Groq** to build
a conversational AI that remembers previous interactions.  
The chat history is stored automatically in Streamlit’s session state, allowing the model
to maintain context across turns and generate more natural, connected responses.
"""

# ----------------------------------------------------------
# Initialize Conversation Memory
# ----------------------------------------------------------
msgs = StreamlitChatMessageHistory(key="langchain_messages")

if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello! How can I assist you today?")

view_messages = st.expander("🧾 View Conversation History")

# ----------------------------------------------------------
# Define Prompt Template
# ----------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly and intelligent AI assistant conversing with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# ----------------------------------------------------------
# Load Groq Model
# ----------------------------------------------------------
groq_api_key = st.sidebar.text_input("🔑 Enter your Groq API Key", type="password")

if not groq_api_key:
    st.info("Please enter your Groq API key to start chatting.")
    st.stop()

# Use the same model from your account list
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    streaming=True,
)

# Combine Model and Conversation Memory
chain_with_history = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# ----------------------------------------------------------
# Display Existing Messages
# ----------------------------------------------------------
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# ----------------------------------------------------------
# Chat Input and Model Response
# ----------------------------------------------------------
if user_input := st.chat_input("Type your message..."):
    st.chat_message("human").write(user_input)
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": user_input}, config)
    st.chat_message("ai").write(response.content)

# ----------------------------------------------------------
# View Session Memory (for transparency)
# ----------------------------------------------------------
with view_messages:
    st.json(st.session_state.langchain_messages)
