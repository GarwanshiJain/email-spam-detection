
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model, vectorizer and dataset
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
df = pd.read_csv("data/processed/cleaned_data.csv")

# Page configuration
st.set_page_config(
    page_title="Spam Message Detection",
    page_icon="📩",
    layout="centered"
)

# Simple black theme
st.markdown("""
<style>
.stApp {
    background-color: #101010;
    color: white;
}

h1, h2, h3 {
    color: white;
}

div.stButton > button {
    background-color: #306B57;
    color: white;
    border-radius: 6px;
    border: none;
}

div.stButton > button:hover {
    background-color: #40866D;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("📩 Spam Message Detection")

st.write(
    "A machine learning application that classifies "
    "messages as Spam or Ham."
)

st.divider()

# Message prediction section
st.subheader("Test a Message")

message = st.text_area(
    "Enter your message:",
    placeholder="Type or paste your message here...",
    height=150
)

if st.button("Detect Message"):

    if not message.strip():
        st.warning("Please enter a message.")

    else:
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)[0]

        probabilities = model.predict_proba(message_tfidf)[0]
        confidence = probabilities[
            list(model.classes_).index(prediction)
        ] * 100

        if prediction == "spam":
            st.error("Result: SPAM")

        else:
            st.success("Result: HAM (Not Spam)")

        st.write(f"Confidence: {confidence:.2f}%")
        st.progress(float(confidence / 100))

        st.caption(
            "Confidence represents the model's estimated "
            "probability for the predicted class. "
            "It is not a guarantee of safety."
        )

st.divider()

# Dataset overview
st.subheader("Dataset Overview")

st.write(
    "Distribution of Spam and Ham messages "
    "in the dataset used for training."
)

label_counts = df["label"].value_counts()

fig, ax = plt.subplots(figsize=(6, 3))

fig.patch.set_facecolor("#101010")
ax.set_facecolor("#101010")

bars = ax.bar(
    ["Ham", "Spam"],
    [label_counts["ham"], label_counts["spam"]],
    color=["#75B798", "#D98272"],
    width=0.5
)

ax.set_ylabel("Number of Messages", color="white")
ax.set_title("Message Distribution", color="white")
ax.tick_params(colors="white")

for spine in ax.spines.values():
    spine.set_color("#555555")

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(int(bar.get_height())),
        ha="center",
        va="bottom",
        color="white"
    )

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

# Footer
st.caption(
    "Developed using Python, Scikit-learn, "
    "TF-IDF and Semi-Supervised Learning."
)
