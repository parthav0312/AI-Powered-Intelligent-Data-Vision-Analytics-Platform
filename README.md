# 🧠 AI-Powered Intelligent Data & Vision Analytics Platform

A unified AI platform that combines **Conversational Data Analytics, Machine Learning, Document Intelligence (RAG), SQL Querying, and Computer Vision** into a single Streamlit application powered by **LangChain, Groq, OpenAI, and PyTorch**.

---

## 📖 Overview

The **AI-Powered Intelligent Data & Vision Analytics Platform** enables users to analyze datasets, query SQL databases, interact with PDF documents, train machine learning models, and perform advanced computer vision tasks through a conversational interface.

Built with **Streamlit**, **LangChain**, **Groq**, and **OpenAI**, the platform provides an intuitive dashboard that simplifies complex AI workflows without requiring users to switch between multiple tools.

---

# 🚀 Features

### 💬 Conversational Data Analytics

* Chat with CSV and Excel datasets using natural language.
* Generate insights without writing SQL or Python code.
* Perform filtering, aggregation, visualization, and trend analysis.

### 🗄️ Natural Language SQL Assistant

* Query SQLite databases using plain English.
* Automatically converts user questions into SQL queries.
* Includes a sample **Chinook** database for testing.

Example:

> *"Show the top 10 customers with the highest total purchases."*

---

### 📄 Document Intelligence (RAG)

* Upload PDF documents.
* Ask context-aware questions.
* Uses LangChain Retrieval-Augmented Generation (RAG).
* Powered by Groq LLMs and OpenAI.

---

### 🤖 Machine Learning Suite

Train and evaluate machine learning models directly from the dashboard.

Supported models include:

* Logistic Regression
* Neural Networks

Features:

* Dataset preprocessing
* Model training
* Performance evaluation
* Accuracy metrics
* Confusion Matrix
* MLflow experiment tracking

---

### 📸 Computer Vision Pro

Comprehensive image processing module featuring:

* Image Enhancement
* Image Filtering
* OCR using EasyOCR
* Object Detection (YOLOv5)
* Face Detection
* Webcam Detection
* Edge Detection
* Real-time Computer Vision

---

### 🧠 Conversational Memory

* Maintains chat history using LangChain Memory.
* Provides context-aware conversations.
* Supports multi-turn interactions.

---

### 📊 Unified Dashboard

All AI modules are accessible from one Streamlit interface.

Navigation includes:

* Chat with SQL
* Chat with Pandas
* Chat with Documents
* Machine Learning
* Computer Vision

---

# 🛠️ Technology Stack

### Frontend

* Streamlit

### AI & LLMs

* LangChain
* Groq API
* OpenAI API

### Data Analytics

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* PyTorch

### Computer Vision

* OpenCV
* YOLOv5
* EasyOCR

### Experiment Tracking

* MLflow

### Database

* SQLite

---

# 📂 Project Structure

```text
streamlit_agent/
│
├── app.py                        # Main Streamlit dashboard
├── chat_pandas_df.py             # Chat with CSV & Excel
├── chat_with_documents.py        # PDF Q&A using RAG
├── chat_with_sql_db.py           # SQL Chat Assistant
├── machine_learning.py           # ML Module
├── computer_vision.py            # Computer Vision Suite
├── Chinook.db                    # Sample SQLite Database
├── requirements.txt              # Dependencies
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Powered-Intelligent-Data-Vision-Analytics-Platform.git

cd AI-Powered-Intelligent-Data-Vision-Analytics-Platform
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv new_venv
new_venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv new_venv
source new_venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Application

```bash
streamlit run app.py
```

Open your browser:

```
http://localhost:8501
```

---

# 🔑 API Configuration

The application supports both **Groq** and **OpenAI**.

| Provider | Purpose                            |
| -------- | ---------------------------------- |
| Groq     | Document Chat, SQL Chat, Data Chat |
| OpenAI   | Conversational AI                  |

Enter your API keys from the Streamlit sidebar after launching the application.

Required formats:

```
Groq:
gsk_xxxxxxxxxxxxxxxxxxxxx

OpenAI:
sk-xxxxxxxxxxxxxxxxxxxx
```

---

# 🗃️ SQL Database

The project includes a sample SQLite database:

```
Chinook.db
```

Replace this file with your own SQLite database if needed.

Example Query:

```
Show the top 10 customers with the highest total purchases.
```

---

# 📈 MLflow (Optional)

To monitor machine learning experiments:

```bash
mlflow ui
```

Open:

```
http://localhost:5000
```

Track:

* Accuracy
* Parameters
* Metrics
* Training Logs
* Model Versions

---

# 💻 System Requirements

| Component       | Requirement               |
| --------------- | ------------------------- |
| Python          | 3.10+                     |
| RAM             | 8 GB Minimum              |
| Recommended RAM | 16 GB                     |
| GPU             | Optional (CUDA Supported) |
| Internet        | Required for API Calls    |

---

# 🧪 Example Use Cases

### 📊 Business Analysts

* Analyze structured datasets using natural language.
* Generate dashboards and reports.

### 📄 Researchers

* Summarize and query academic PDFs.
* Extract insights from documents.

### 🤖 Data Scientists

* Train machine learning models.
* Compare model performance.
* Track experiments using MLflow.

### 👨‍💻 Developers

* Build intelligent AI workflows.
* Integrate conversational analytics with computer vision.

---

# 🧰 Troubleshooting

| Issue                        | Solution                               |
| ---------------------------- | -------------------------------------- |
| ModuleNotFoundError          | Run `pip install -r requirements.txt`  |
| Invalid Groq API Key         | Generate a new key from Groq Console   |
| OpenAI Authentication Error  | Update your API key                    |
| cv2.error                    | Use odd Gaussian Blur values (3, 5, 7) |
| Duplicate Streamlit Elements | Upgrade Streamlit                      |
| `ValueError: y should be 1D` | Select a categorical target column     |

---

# 🌟 Highlights

* Conversational AI for Data Analysis
* SQL Query Assistant
* PDF Question Answering (RAG)
* Machine Learning Pipeline
* Computer Vision Toolkit
* YOLOv5 Object Detection
* OCR using EasyOCR
* MLflow Integration
* LangChain Memory
* Streamlit Dashboard
* Groq & OpenAI Support

---

# 📜 License

This project is intended for educational, research, and portfolio purposes.

---

## ⭐ If you found this project useful, consider giving it a Star on GitHub!
