import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App UI
st.set_page_config(page_title="SMS Spam Detector", layout="centered")
st.title("📱 SMS Spam Classifier")

user_input = st.text_area("Enter your SMS message here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Vectorize input and predict
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][prediction]

        if prediction == 1:
            st.error(f"🚫 This message is predicted as **SPAM** with {prob:.2%} confidence.")
        else:
            st.success(f"✅ This message is predicted as **HAM** (not spam) with {prob:.2%} confidence.")
