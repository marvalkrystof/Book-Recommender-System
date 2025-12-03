import streamlit as st
import numpy as np
import pandas as pd
import faiss
from utils import search_books, rerank_books

# Load FAISS index
index = faiss.read_index("data/book_faiss.index")
# Load Dataset
data_df = pd.read_csv("data/processed_data_2.3.csv")

TOP_K = 50


st.title("Book Recommender System")

st.write("Enter your preferences to get book recommendations")

# Text input
user_input = st.text_input("What kind of books are you looking for?", 
                           placeholder="e.g., fantasy novels with strong female characters")

# Display the input
if user_input:
    st.write(f"You're looking for: {user_input}")
    
    # Get recommendations
    st.subheader("Recommendations")
    with st.spinner("Finding books for you..."):
        results = search_books(user_input, index, data_df, top_k=TOP_K)[['Title', 'Authors', 'Description']]

    with st.spinner("Reranking.."):
        results_reranked = rerank_books(user_input, results)[['Title', 'Authors', 'Description']]

    # Display results
    st.dataframe(results_reranked, use_container_width=True)