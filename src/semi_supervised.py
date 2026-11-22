import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

df["cleaned_message"] = df["cleaned_message"].fillna("").astype(str)

X = df["cleaned_message"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF feature extraction
vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Select only 30% of training data as labeled
labeled_indices, unlabeled_indices = train_test_split(
    np.arange(len(y_train)),
    train_size=0.3,
    random_state=42,
    stratify=y_train
)

# Mark remaining data as unlabeled (-1)
y_semi = np.array(y_train, dtype=object)
y_semi[unlabeled_indices] = -1

# Create base model
base_model = LogisticRegression(max_iter=1000)

# Self-training model
model = SelfTrainingClassifier(
    estimator=base_model,
    threshold=0.60
)

# Train model
model.fit(X_train_tfidf, y_semi)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("SEMI-SUPERVISED MODEL RESULTS")
print("----------------------------------------")

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nInitially labeled samples:", len(labeled_indices))
print("Initially unlabeled samples:", len(unlabeled_indices))
print("Self-training rounds:", model.n_iter_)
print("Pseudo-labeled samples:", np.sum(model.labeled_iter_ > 0))


# Save trained model and vectorizer
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")