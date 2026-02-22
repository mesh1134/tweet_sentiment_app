# 🐦 Twitter Sentiment Analysis Engine (Deep Learning)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tweetsentimentapp-sqpyaywpahcfwtpvfnaxb9.streamlit.app/)

## 📌 Project Overview
This project is a Natural Language Processing (NLP) application that classifies the sentiment of tweets into three categories: Positive, Neutral, and Negative. By leveraging Deep Learning, the model understands contextual nuances and complex sentence structures better than traditional bag-of-words approaches.

## 🚀 Live Web Application
The predictive model is deployed as an interactive web application using Streamlit.
* **Try it live here:** https://tweetsentimentapp-sqpyaywpahcfwtpvfnaxb9.streamlit.app/

## 🧠 Technical Architecture
* **Deep Learning Model:** Architected a Bidirectional Long Short-Term Memory (Bi-LSTM) neural network using TensorFlow and Keras. The bidirectional wrapper allows the model to process sequences in both directions, capturing advanced context (e.g., sarcasm or double negatives).
* **Optimization & Regularization:** Utilized an Embedding layer mapped to a spatial dropout (`SpatialDropout1D`) and recurrent dropout techniques to prevent overfitting on the training data.
* **Robust Preprocessing:** Mitigated training-serving skew by implementing a unified, regex-based text preprocessing pipeline. The exact same cleaning sequence (removing URLs, mentions, hashtags, and punctuation) is applied during both model training and live Streamlit user inference.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** Pandas, NumPy, Scikit-Learn, Joblib
* **Deployment:** Streamlit Community Cloud

## 💻 How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `python -m streamlit run app.py`
