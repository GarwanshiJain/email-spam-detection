import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Handle missing values
df["cleaned_message"] = df["cleaned_message"].fillna("").astype(str)

# Extract cleaned messages
messages = df["cleaned_message"]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Convert text into numerical features
X = vectorizer.fit_transform(messages)

# Display results
print("TF-IDF matrix shape:", X.shape)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Extract non-zero values from the first message
first_message = X[0]

words = feature_names[first_message.indices]
scores = first_message.data

# Display words and their TF-IDF scores
print("\nWords and TF-IDF scores of the first message:")

for word, score in zip(words, scores):
    print(word, ":", round(score, 4))