import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_customer_support_tickets.csv")
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data
df = load_data()

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_text'])

# Streamlit UI
st.title("🤖 Customer Support Chatbot")
st.write("Ask your question below:")

user_query = st.text_input("Your Query:")

if user_query:
    cleaned_query = clean_text(user_query)
    query_vec = vectorizer.transform([cleaned_query])
    similarity = cosine_similarity(query_vec, X)
    best_match = np.argmax(similarity)
    response = df.iloc[best_match]['cleaned_response']
    st.markdown("**Chatbot Response:**")
    st.write(response)
