import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

df["cleaned_message"] = df["cleaned_message"].fillna("").astype(str)

# Separate messages and labels
X = df["cleaned_message"]
y = df["label"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Display dataset information
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

print("\nData preparation completed successfully!")