print("I AM STARTING...")
from twitter_sentiment_main import clean_text

import joblib
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 1. Load the Brains
@st.cache_resource
def load_artifacts():
    try:
        model = load_model('sentiment_model.keras')
        tokenizer = joblib.load('tokenizer.joblib')
        return model, tokenizer
    except Exception as e:
        return None, None


model, tokenizer = load_artifacts()

if model is None:
    st.error("Error: Could not load 'sentiment_model.keras' or 'tokenizer.joblib'. Did you run the notebook?")
    st.stop()


def decode_sentiment(score):
    labels = ["Negative 😠", "Neutral 😐", "Positive 😃"]

    # NEW CODE (Trust the highest score, no matter what)
    label = labels[np.argmax(score)]
    confidence = np.max(score)

    return label, confidence


# 3. The App Interface
st.title("🐦 Pro Twitter Sentiment Analyzer")
st.markdown("Enter a tweet to analyze its sentiment.")

user_input = st.text_area("Tweet:", placeholder="The new update is absolutely terrible...")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please type something first!")
    else:
        # A. Preprocess (The Missing Link!)
        cleaned_input = clean_text(user_input)

        # B. Tokenize & Pad
        seq = tokenizer.texts_to_sequences([cleaned_input])
        padded = pad_sequences(seq, maxlen=40, padding='post')

        # C. Predict
        prediction = model.predict(padded)[0]
        sentiment, confidence = decode_sentiment(prediction)

        # D. Display
        st.divider()
        st.subheader(f"Sentiment: {sentiment}")
        st.caption(f"Confidence: {confidence:.2%}")

        with st.expander("Debug Details"):
            st.text(f"Raw Input: {user_input}")
            st.text(f"Cleaned:   {cleaned_input}")  # Verify it works
            st.write(f"Probabilities: {prediction}")
