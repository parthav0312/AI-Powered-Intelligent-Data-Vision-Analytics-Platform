from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------
# Supported file formats
# ----------------------------------------------------------
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

# ----------------------------------------------------------
# Utility: Load data
# ----------------------------------------------------------
@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    """Read uploaded file into a DataFrame"""
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except Exception:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    st.error(f"Unsupported file format: {ext}")
    return None


# ----------------------------------------------------------
# Streamlit page config
# ----------------------------------------------------------
st.set_page_config(page_title="🦜 Chat with pandas DataFrame (Groq)", page_icon="🦜")
st.title("🦜 LangChain + Groq : Chat with pandas DataFrame")

# ----------------------------------------------------------
# Step 1 – API key input
# ----------------------------------------------------------
groq_api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password")
if not groq_api_key:
    st.info("Please enter your **Groq API key** in the sidebar to begin.")
    st.stop()

# ----------------------------------------------------------
# Step 2 – File upload
# ----------------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV or Excel File",
    type=list(file_formats.keys()),
    help="Supported formats: CSV, XLS, XLSX, XLSM, XLSB",
)
if not uploaded_file:
    st.info("Please upload a CSV or Excel file to analyze.")
    st.stop()

# ----------------------------------------------------------
# Step 3 – Load data and summarize columns
# ----------------------------------------------------------
df = load_data(uploaded_file)
if df is None:
    st.stop()

st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
st.dataframe(df.head())

# Auto-detect column types
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

st.sidebar.markdown("### 📊 Data Summary")
st.sidebar.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
st.sidebar.write(f"**Numeric Columns:** {', '.join(numeric_cols) if numeric_cols else 'None'}")
st.sidebar.write(f"**Categorical Columns:** {', '.join(categorical_cols) if categorical_cols else 'None'}")

# ----------------------------------------------------------
# Conversation state
# ----------------------------------------------------------
if "messages" not in st.session_state or st.sidebar.button("🧹 Clear Conversation"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi there! I'm ready to analyze your data. Ask me anything about it or request a chart."}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ----------------------------------------------------------
# Chat interaction
# ----------------------------------------------------------
if prompt := st.chat_input("Ask a question about your data…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # ------------------------------------------------------
    # Initialize Groq model
    # ------------------------------------------------------
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        streaming=False,
    )

    # ------------------------------------------------------
    # Create DataFrame agent
    # ------------------------------------------------------
    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # better text output
        handle_parsing_errors=True,
        allow_dangerous_code=True,
    )

    # ------------------------------------------------------
    # Generate AI response and detect if chart is requested
    # ------------------------------------------------------
    with st.chat_message("assistant"):
        response = pandas_df_agent.run(prompt)
        ai_output = str(response).strip()
        if not ai_output:
            ai_output = "⚠️ The model ran successfully but didn't return any text output."

        # Basic chart detection keywords
        chart_keywords = ["plot", "chart", "graph", "visualize", "draw"]
        if any(k in prompt.lower() for k in chart_keywords):
            try:
                st.write("📈 Generating visualization based on your data…")
                fig, ax = plt.subplots(figsize=(6, 4))
                # Simple example heuristic
                if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                    df.groupby(categorical_cols[0])[numeric_cols[0]].mean().plot(kind="bar", ax=ax)
                    plt.title(f"{numeric_cols[0]} by {categorical_cols[0]}")
                elif len(numeric_cols) >= 2:
                    df.plot(x=numeric_cols[0], y=numeric_cols[1], kind="line", ax=ax)
                    plt.title(f"{numeric_cols[1]} vs {numeric_cols[0]}")
                elif len(numeric_cols) == 1:
                    df[numeric_cols[0]].plot(kind="hist", ax=ax)
                    plt.title(f"Distribution of {numeric_cols[0]}")
                else:
                    st.warning("No suitable numeric/categorical columns found for chart.")
                    fig = None

                if fig:
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"⚠️ Error while generating chart: {e}")

        # Show text answer in chat
        st.session_state["messages"].append({"role": "assistant", "content": ai_output})
        st.markdown(ai_output)
