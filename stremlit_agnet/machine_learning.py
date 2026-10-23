import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Streamlit Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="Machine Learning | Logistic + PyTorch + MLflow", page_icon="🧠")
st.title("🧠 Machine Learning Models – Logistic Regression & Neural Network")

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
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Select Target Column (what to predict)", df.columns)
    if df[target_col].nunique() > 10:
        st.warning(f"⚠️ The selected target '{target_col}' has {df[target_col].nunique()} unique values — it may not be suitable for classification.")

    feature_cols = st.multiselect(
        "🧮 Select Feature Columns (inputs)",
        [col for col in df.columns if col != target_col],
        default=[]
    )

    if feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        # Handle non-numeric features automatically
        X = pd.get_dummies(X, drop_first=True)

        # Properly encode categorical target
        if y.dtype == "object" or str(y.dtype) == "category":
            y = y.astype("category").cat.codes  # convert to numeric labels

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Scale numeric data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Select model type
        model_type = st.radio("🧠 Choose Model Type", ["Logistic Regression (Sklearn)", "Neural Network (PyTorch)"])

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

                st.success(f"✅ Logistic Regression Trained! Accuracy: **{acc*100:.2f}%**")

        else:
            # ------------------------------------------------------
            # Define Simple Neural Network
            # ------------------------------------------------------
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

            # ------------------------------------------------------
            # Training Parameters
            # ------------------------------------------------------
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

                st.success(f"✅ Neural Network Trained! Accuracy: **{acc*100:.2f}%**")

        # ------------------------------------------------------
        # Confusion Matrix + Report
        # ------------------------------------------------------
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
        st.info("📁 MLflow has logged this experiment. Run `mlflow ui` in terminal and open http://localhost:5000 to view.")
    else:
        st.info("Please select at least one feature column to train the model.")
else:
    st.warning("Please upload a CSV file to begin.")
