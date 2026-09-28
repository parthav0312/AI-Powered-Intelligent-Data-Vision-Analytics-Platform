# ===========================
# Unified Streamlit Super-App
# ===========================
# Modules merged:
# 1) Chat with SQL DB (LangChain + Groq)
# 2) Chat with Pandas (LangChain + Groq)
# 3) Chat with Documents (PDF RAG, LangChain + Groq)
# 4) Machine Learning (Logistic Regression)
# 5) Computer Vision (Enhancement + Live camera detection)

import os
import time
import tempfile
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import mlflow
import mlflow.sklearn
import mlflow.pytorch


import streamlit as st

# LangChain & Groq (LLM)
from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# RAG / Documents
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler

# Classic ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ----------------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="Intelligent Data & Vision Analytics Platform", page_icon="🤖")
st.title("🤖 Intelligent Data & Vision Analytics Platform")
st.caption(
    "An integrated Streamlit-based platform unifying conversational analytics, "
    "machine learning experimentation, and advanced computer vision workflows — "
    "powered by LangChain, Groq, PyTorch, OpenCV, and MLflow."
)

# ==========================================================
# 1) Chat with SQL DB
# ==========================================================
def chat_with_sql_app():
    st.header("💬 Chat with SQL Database (LangChain + Groq)")

    # Groq key
    groq_api_key = st.sidebar.text_input("🔑 Groq API Key", type="password", key="sql_groq_key")
    if not groq_api_key:
        st.info("Enter your Groq API key to continue.")
        st.stop()

    # Example: Using Chinook.db (if present)
    st.markdown("This page can be wired to your SQL DB (e.g., SQLite `Chinook.db`).")
    db_path = st.text_input("SQLite DB path", value="Chinook.db")

    # Minimal chat loop with a placeholder answer to show structure.
    # You can plug in LangChain SQLDatabase + SQL agent here if desired.
    # (Kept light because your original SQL file can vary.)
    if "sql_msgs" not in st.session_state:
        st.session_state["sql_msgs"] = [{"role": "assistant", "content": "Hi! Ask me about your database."}]

    # render
    for m in st.session_state["sql_msgs"]:
        st.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("Ask a SQL question…"):
        st.chat_message("user").write(prompt)

        # You can implement a LangChain SQL agent here with Groq LLM:
        # from langchain_community.utilities import SQLDatabase
        # from langchain.agents.agent_toolkits import create_sql_agent
        # db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        # llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0)
        # agent = create_sql_agent(llm=llm, db=db, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        # response = agent.run(prompt)
        #
        # For now, a placeholder response (to keep structure intact):
        response = f"(demo) I would run a SQL query on {db_path} matching: {prompt}"

        st.session_state["sql_msgs"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)


# ==========================================================
# 2) Chat with Pandas DataFrame
# ==========================================================
def chat_with_pandas_app():
    st.header("🦜 Chat with Pandas DataFrame (LangChain + Groq)")

    groq_api_key = st.sidebar.text_input("🔑 Groq API Key", type="password", key="pandas_groq_key")
    if not groq_api_key:
        st.info("Enter your Groq API key to continue.")
        st.stop()

    # Upload CSV/Excel
    file_formats = {"csv": pd.read_csv, "xls": pd.read_excel, "xlsx": pd.read_excel, "xlsm": pd.read_excel, "xlsb": pd.read_excel}
    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV/Excel", type=list(file_formats.keys()), key="pandas_uploader")

    if not uploaded_file:
        st.info("Upload a file to chat with your data.")
        st.stop()

    # Load data
    ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    df = file_formats[ext](uploaded_file)
    st.dataframe(df.head(), use_container_width=True)

    # Chat messages
    if "pandas_msgs" not in st.session_state or st.sidebar.button("🧹 Clear conversation", key="clear_pandas_conv"):
        st.session_state["pandas_msgs"] = [{"role": "assistant", "content": "Hi! Ask me about your dataset."}]

    for msg in st.session_state["pandas_msgs"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat loop
    if prompt := st.chat_input("Ask a question about your data…"):
        st.chat_message("user").write(prompt)

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0,
            streaming=False,
        )

        # Better with ZERO_SHOT_REACT_DESCRIPTION to ensure textual outputs
        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
        )

        response = pandas_df_agent.run(prompt)
        ai_output = str(response).strip()
        if not ai_output:
            ai_output = "⚠️ The model ran successfully but returned no text."

        st.session_state["pandas_msgs"].append({"role": "assistant", "content": ai_output})
        st.chat_message("assistant").write(ai_output)


# ==========================================================
# 3) Chat with Documents (PDF RAG)
# ==========================================================
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


def chat_with_documents_app():
    st.header("📄 Chat with Documents (Groq LLM)")

    groq_api_key = st.sidebar.text_input("🔑 Groq API Key", type="password", key="docs_groq_key")
    if not groq_api_key:
        st.info("Enter your Groq API key to continue.")
        st.stop()

    uploaded_files = st.sidebar.file_uploader("📂 Upload PDF files", type=["pdf"], accept_multiple_files=True, key="docs_uploader")
    if not uploaded_files:
        st.info("Upload at least one PDF to start the chat.")
        st.stop()

    @st.cache_resource(ttl="1h")
    def build_retriever(files):
        docs = []
        tmp = tempfile.TemporaryDirectory()
        for f in files:
            path = os.path.join(tmp.name, f.name)
            with open(path, "wb") as out:
                out.write(f.getvalue())
            docs.extend(PyPDFLoader(path).load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = DocArrayInMemorySearch.from_documents(split_docs, embeddings)
        return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})

    retriever = build_retriever(uploaded_files)

    msgs = StreamlitChatMessageHistory(key="docs_messages")
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        streaming=True,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=False
    )

    # Render history
    for m in msgs.messages:
        st.chat_message(m.type).write(m.content)

    # Chat
    if user_query := st.chat_input("Ask a question about your PDFs…"):
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[stream_handler])
            if not response:
                response = "⚠️ The model didn’t return any text."
            st.write(response)


# ==========================================================
# 4) Machine Learning – Logistic Regression
# ==========================================================
def machine_learning_app():
    st.header("🧠 Machine Learning Models – Logistic Regression & Neural Network")

    st.markdown("""
    Upload a **CSV dataset**, select your **target column**, and this app will:
    1. Split the data into train/test sets  
    2. Scale the features  
    3. Train **either Logistic Regression (Sklearn)** or a **Neural Network (PyTorch)**  
    4. Log all experiments automatically with **MLflow**  
    5. Display metrics and confusion matrix  
    ---
    """)

    # ----------------------------------------------------------
    # File Upload
    # ----------------------------------------------------------
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"], key="ml_csv_upload")

    if not uploaded_file:
        st.warning("Please upload a CSV file to begin.")
        return

    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    target_col = st.selectbox("🎯 Select Target Column (what to predict)", df.columns, key="ml_target_col")
    if df[target_col].nunique() > 10:
        st.warning(f"⚠️ The selected target '{target_col}' has {df[target_col].nunique()} unique values — it may not be suitable for classification.")

    feature_cols = st.multiselect(
        "🧮 Select Feature Columns (inputs)",
        [col for col in df.columns if col != target_col],
        default=[],
        key="ml_feature_cols"
    )

    if not feature_cols:
        st.info("Please select at least one feature column to train the model.")
        return

    X = df[feature_cols]
    y = df[target_col]

    # Handle non-numeric features automatically
    X = pd.get_dummies(X, drop_first=True)

    # Encode categorical target
    if y.dtype == "object" or str(y.dtype) == "category":
        y = y.astype("category").cat.codes

    # Check for single-class target early
    if len(np.unique(y)) < 2:
        st.error("❌ Target column contains only one unique class. Please select another column.")
        return

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Guard again post-split
    if len(np.unique(y_train)) < 2:
        st.error("❌ Training data has only one class after splitting. Please use a larger or more balanced dataset.")
        return

    # Scale numeric data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------------------------------------
    # Select model type
    # ----------------------------------------------------------
    model_type = st.radio("🧠 Choose Model Type", ["Logistic Regression (Sklearn)", "Neural Network (PyTorch)"], key="ml_model_choice")

    # Initialize MLflow experiment
    mlflow.set_experiment("Streamlit_ML_Experiments")

    if model_type == "Logistic Regression (Sklearn)":
        with mlflow.start_run(run_name="Logistic_Regression"):
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)

            # ✅ Log metrics and model
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

            st.success(f"✅ Logistic Regression Trained Successfully! Accuracy: **{acc*100:.2f}%**")

    else:
        # ----------------------------------------------------------
        # Define Simple Neural Network
        # ----------------------------------------------------------
        class Net(nn.Module):
            def __init__(self, input_dim):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x

        # Convert data to tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(np.array(y_train).reshape(-1, 1), dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(np.array(y_test).reshape(-1, 1), dtype=torch.float32).to(device)

        # ----------------------------------------------------------
        # Training Parameters
        # ----------------------------------------------------------
        model = Net(X_train.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = 200

        with mlflow.start_run(run_name="Neural_Network"):
            mlflow.log_param("model_type", "PyTorch_NN")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("learning_rate", 0.001)

            # Training Loop
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                preds = (model(X_test_tensor) > 0.5).float()
                acc = (preds.eq(y_test_tensor).sum().item()) / len(y_test_tensor)

            # ✅ Log metrics and model
            mlflow.log_metric("accuracy", acc)
            mlflow.pytorch.log_model(model, "model")

            st.success(f"✅ Neural Network Trained Successfully! Accuracy: **{acc*100:.2f}%**")

    # ----------------------------------------------------------
    # Confusion Matrix + Report
    # ----------------------------------------------------------
    st.subheader("📉 Confusion Matrix")
    if model_type == "Logistic Regression (Sklearn)":
        y_pred_final = y_pred
    else:
        y_pred_final = preds.cpu().numpy()
        y_pred_final = np.round(y_pred_final).astype(int)

    cm = confusion_matrix(y_test, y_pred_final)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("📊 Classification Report")
    st.text(classification_report(y_test, y_pred_final))

    st.info("📁 MLflow has logged this experiment. Run `mlflow ui` and open http://localhost:5000 to view runs.")



# ==========================================================
# 5) Computer Vision Suite
# ==========================================================
def computer_vision_app():
    st.header("📸 Computer Vision  OpenCV + PyTorch + MLflow")
    st.caption("Advanced Image Processing • Object Detection • OCR • Experiment Tracking • Live Camera")

    # ----------------------------------------------------------
    # Sidebar Controls
    # ----------------------------------------------------------
    with st.sidebar:
        st.header("🔧 Configuration")
        mode = st.radio(
            "Select Task",
            [
                "Image Enhancement",
                "Object Detection (YOLOv5)",
                "OCR Text Extraction",
                "Face Detection",
                "Live Camera Detection",
            ],
            index=0,
            key="cv_mode"
        )
        mlflow_logging = st.checkbox("Enable MLflow Tracking", value=False, key="cv_mlflow")

    # ----------------------------------------------------------
    # MLflow Setup
    # ----------------------------------------------------------
    if mlflow_logging:
        mlflow.set_experiment("Computer_Vision")
        run = mlflow.start_run(run_name=mode)
        start_time = time.time()

    # ----------------------------------------------------------
    # File Upload (not needed for Live mode)
    # ----------------------------------------------------------
    uploaded_file = None
    if mode != "Live Camera Detection":
        uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"], key="cv_upload")

        if uploaded_file is None:
            st.info("Please upload an image to begin.")
            st.stop()

        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        col1.image(image_rgb, caption="Original Image", use_container_width=True)

    # ----------------------------------------------------------
    # Mode 1 – Image Enhancement
    # ----------------------------------------------------------
    if mode == "Image Enhancement":
        st.subheader("🧠 Image Enhancement")
        blur = st.slider("Gaussian Blur", 1, 15, 3, step=2, key="cv_blur")
        brightness = st.slider("Brightness Adjustment", -50, 50, 0, key="cv_bright")
        contrast = st.slider("Contrast Adjustment", 1.0, 3.0, 1.0, key="cv_contrast")

        enhanced = cv2.GaussianBlur(image_rgb, (blur, blur), 0)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=brightness)
        col2.image(enhanced, caption="Enhanced Image", use_container_width=True)

        if mlflow_logging:
            mlflow.log_param("blur", blur)
            mlflow.log_param("contrast", contrast)
            mlflow.log_param("brightness", brightness)

    # ----------------------------------------------------------
    # Mode 2 – Object Detection (YOLOv5)
    # ----------------------------------------------------------
    elif mode == "Object Detection (YOLOv5)":
        st.subheader("🎯 Object Detection using YOLOv5")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to(device)
        results = model(image_rgb)
        detected = np.squeeze(results.render())
        col2.image(detected, caption="Detected Objects", use_container_width=True)
        st.dataframe(results.pandas().xyxy[0])

        if mlflow_logging:
            mlflow.log_param("model", "yolov5s")
            mlflow.log_metric("detections", len(results.pandas().xyxy[0]))

    # ----------------------------------------------------------
    # Mode 3 – OCR Text Extraction
    # ----------------------------------------------------------
    elif mode == "OCR Text Extraction":
        st.subheader("📄 Optical Character Recognition (OCR)")
        reader = easyocr.Reader(["en"])
        result = reader.readtext(image_rgb)
        ocr_img = image_rgb.copy()

        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            cv2.rectangle(ocr_img, tuple(map(int, top_left)), tuple(map(int, bottom_right)), (0, 255, 0), 2)
            cv2.putText(ocr_img, text, (int(top_left[0]), int(top_left[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        col2.image(ocr_img, caption="OCR Output", use_container_width=True)
        extracted_text = [text for (_, text, _) in result]
        st.write("**Extracted Text:**", extracted_text)

        if mlflow_logging:
            mlflow.log_param("ocr_texts", len(extracted_text))
            mlflow.log_text("\n".join(extracted_text), "ocr_output.txt")

    # ----------------------------------------------------------
    # Mode 4 – Face Detection
    # ----------------------------------------------------------
    elif mode == "Face Detection":
        st.subheader("😎 Face Detection (Haar Cascade)")
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_img = image_rgb.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        col2.image(face_img, caption=f"Detected {len(faces)} Face(s)", use_container_width=True)

        if mlflow_logging:
            mlflow.log_metric("faces_detected", len(faces))

    # ----------------------------------------------------------
    # Mode 5 – Live Camera Detection
    # ----------------------------------------------------------
    elif mode == "Live Camera Detection":
        st.subheader("🎥 Real-time Object Detection (YOLOv5 + OpenCV)")

        run_live = st.button("Start Camera", key="cv_start_camera")

        if run_live:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to(device)
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            prev_time = time.time()

            if mlflow_logging:
                mlflow.log_param("model", "yolov5s_realtime")

            st.session_state["camera_running"] = True
            stop_stream = st.sidebar.button("Stop Camera", key="cv_stop_camera")

            while cap.isOpened() and st.session_state.get("camera_running", True):
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Unable to access camera.")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame)
                annotated = np.squeeze(results.render())

                fps = 1.0 / (time.time() - prev_time)
                prev_time = time.time()

                stframe.image(annotated, channels="RGB", use_container_width=True)
                st.sidebar.info(f"⚡ FPS: {fps:.2f}")

                if stop_stream:
                    st.session_state["camera_running"] = False
                    break

            cap.release()
            st.success("✅ Camera stopped.")

    # ----------------------------------------------------------
    # MLflow End Run
    # ----------------------------------------------------------
    if mlflow_logging:
        mlflow.log_metric("runtime_sec", round(time.time() - start_time, 2))
        mlflow.end_run()

# ==========================================================
# Sidebar Navigation
# ==========================================================
st.sidebar.title("🧭 Navigation")
choice = st.sidebar.radio(
    "Choose a module:",
    [
        "Chat with SQL DB",
        "Chat with Pandas",
        "Chat with Documents",
        "Machine Learning",
        "Computer Vision",
    ],
    key="main_nav_radio",
)

if choice == "Chat with SQL DB":
    chat_with_sql_app()
elif choice == "Chat with Pandas":
    chat_with_pandas_app()
elif choice == "Chat with Documents":
    chat_with_documents_app()
elif choice == "Machine Learning":
    machine_learning_app()
elif choice == "Computer Vision":
    computer_vision_app()
