import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler

# ----------------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="Chat with Documents – Groq", page_icon="🦜")
st.title("🦜 Chat with Documents (Groq LLM)")

"""
Upload PDF files and chat with their content using **Groq's ultra-fast LLMs**.  
This version uses the **llama-3.3-70b-versatile** model via the **Groq API**, providing exceptional reasoning capability, long-context understanding, and high-quality responses.
"""

# ----------------------------------------------------------
# Sidebar – API Key & File Upload
# ----------------------------------------------------------
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

if not groq_api_key:
    st.info("Please enter your Groq API key to continue.")
    st.stop()

uploaded_files = st.sidebar.file_uploader(
    "📄 Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Upload at least one PDF document to start chatting.")
    st.stop()

# ----------------------------------------------------------
# Document Loading & Embedding
# ----------------------------------------------------------
@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploaded_files:
        temp_path = os.path.join(temp_dir.name, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(split_docs, embeddings)
    return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})

retriever = configure_retriever(uploaded_files)

# ----------------------------------------------------------
# Setup Memory and Chat History
# ----------------------------------------------------------
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear Conversation"):
    msgs.clear()
    msgs.add_ai_message("Hi! I’m ready to answer questions about your PDFs.")

# ----------------------------------------------------------
# Callback Handlers
# ----------------------------------------------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# ----------------------------------------------------------
# Load Groq Model
# ----------------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    streaming=True,
)

# ----------------------------------------------------------
# Build Conversational Retrieval Chain
# ----------------------------------------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

# ----------------------------------------------------------
# Display Conversation
# ----------------------------------------------------------
avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# ----------------------------------------------------------
# Chat Interaction
# ----------------------------------------------------------
if user_query := st.chat_input("Ask a question about your documents..."):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[stream_handler])

        # 🧹 Remove user input if AI echoes it
        ai_output = str(response).strip()
        if ai_output.lower().startswith(user_query.lower()):
            ai_output = ai_output[len(user_query):].strip(" :.-\n")

        st.markdown(ai_output)
