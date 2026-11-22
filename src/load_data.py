import pandas as pd

data_path = "data/raw/SMSSpamCollection"

df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "message"])

print(df.head())

print("Dataset shape:", df.shape)
print("Total messages:", df.shape[0])
print(df["label"].value_counts())
print(df.info())
print(df.isnull().sum())

df["message_length"] = df["message"].apply(len)

print(df[["label", "message_length"]].head())

print("Average message length:", df["message_length"].mean())

# Visualize message length distribution

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))

sns.histplot(data=df, x="message_length", hue="label", bins=40, kde=True)

plt.title("Message Length Distribution: Spam vs Ham")
plt.xlabel("Message Length (Characters)")
plt.ylabel("Number of Messages")

plt.show()

# View sample Spam and Ham messages
print("\n--- Spam Messages ---")
print(df[df["label"] == "spam"]["message"].head(5))

print("\n--- Ham Messages ---")
print(df[df["label"] == "ham"]["message"].head(5))