import pandas as pd
import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("data.csv")

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['cleaned'] = data['text'].apply(preprocess)

# ---------------------------
# Balance Dataset (if needed)
# ---------------------------
majority = data[data.label == data.label.value_counts().idxmax()]
minority = data[data.label != data.label.value_counts().idxmax()]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

data = pd.concat([majority, minority_upsampled])

# ---------------------------
# TF-IDF (Improved)
# ---------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.9
)

X = vectorizer.fit_transform(data['cleaned'])
y = data['label']

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Train Model
# ---------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# ---------------------------
# Accuracy
# ---------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Cyberbullying Detection App")

st.write(f"Model Accuracy: {round(accuracy * 100, 2)}%")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = preprocess(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector).max()

        if prediction.lower() == "bullying":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.write(f"Confidence: {round(probability * 100, 2)}%")