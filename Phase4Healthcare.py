# AI_healthcare_diagnostics.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load and preprocess the dataset
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)

    if 'diagnosis' not in df.columns:
        raise ValueError("Dataset must contain a 'diagnosis' column as the target.")

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, le

# 2. Train model and evaluate performance
def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

    print(f"Accuracy: {acc:.2f}")
    print(f"AUC Score : {auc:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return model

# 3. Make a prediction
def make_prediction(model, sample):
    sample = np.array(sample).reshape(1, -1)
    prediction = model.predict(sample)
    return prediction

# === Main Execution ===
if _name_ == "_main_":
    filepath = 'your_data.csv'  # Replace with your actual CSV filename
    X, y, label_encoder = load_and_preprocess_data(filepath)
    model = train_and_evaluate_model(X, y)

    # Example prediction
    sample_input = X[0]  # Using the first record as a sample
    result = make_prediction(model, sample_input)
    print("Sample Prediction:", label_encoder.inverse_transform(result))