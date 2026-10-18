import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATASET_PATH =  "training/dataset/training_logs.csv"
MODEL_PATH = "models/bert_classifier_model.joblib"


# Code used to build the inference model.
def save_bert_model():
    df = pd.read_csv(DATASET_PATH)
    df_non_legacy = df[df['source']!='LegacyCRM']

    model = SentenceTransformer('all-MiniLM-L6-v2')
    filtered_embeddings = model.encode(
        df_non_legacy['log_message'].tolist(),
        batch_size=64, show_progress_bar=True
    )

    X = filtered_embeddings
    y = df_non_legacy['target_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    joblib.dump(clf, MODEL_PATH)


# Bert processor code.
model = SentenceTransformer('all-MiniLM-L6-v2')
clf = joblib.load(MODEL_PATH)

def classify_with_bert(log_message):
    embeddings = model.encode([log_message])
    probabilities = clf.predict_proba(embeddings)[0]
    if max(probabilities) < 0.5:
        return "Unclassified"
    predicted_label = clf.predict(embeddings)[0]

    return predicted_label


if __name__ == "__main__":
    # save_bert_model()
    logs = [
        "System crashed due to drivers errors when restarting the server",
        "chat, are we cooked?",
        "Multiple login failures occurred on user 6454 account",
    ]

    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)
