import re
import pandas as pd

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Load dataset
data_path = "data/raw/SMSSpamCollection"

df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "message"])


# Apply text cleaning
df["cleaned_message"] = df["message"].apply(clean_text)


# Save cleaned dataset
df.to_csv("data/processed/cleaned_data.csv", index=False)


# Display messages
print(df[["message", "cleaned_message"]].head(10))

print("\nCleaned dataset saved successfully!")