import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Loading IMDB dataset word index
word_index = imdb.get_word_index()

# Loading model
@st.cache_resource
def load_trained_model():
    return load_model('simpleRNN.h5')

model = load_trained_model()

# Preprocessing user input
def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [1] 
    for word in words:
        encoded = word_index.get(word, 2) + 3
     
        if encoded >= 10000:
            encoded = 2 

        encoded_review.append(encoded)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review

# Prediction function
def prediction(review):
    preprocessed_input = preprocess_text(review)
    pred = model.predict(preprocessed_input)
    if pred[0][0] > 0.5:
        result = "Positive"
    else:
        result = "Negative"
    return pred[0][0], result

# Streamlit App
st.set_page_config(page_title="ReviewSense", page_icon=":movie_camera:", layout="centered")

st.title("ReviewSense - Sentiment Analyzer")
st.markdown("Enter a movie review below to analyze its sentiment using a SimpleRNN model trained on IMDB dataset.")

# Text input
review_input = st.text_area(
    "Enter your movie review here:",
    height=150,
    placeholder="Type or paste your movie review here..."
)

# Prediction button
if st.button("Analyze Sentiment", type="primary"):
    if review_input and review_input.strip():
        with st.spinner("Analyzing..."):
            score, result = prediction(review_input)
        
        # Display results
        st.subheader("Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", result)
        with col2:
            st.metric("Confidence", f"{score:.2%}")
        
        # Progress bar
        st.progress(float(score))
        

    else:
        st.warning("Please enter a review to analyze.")