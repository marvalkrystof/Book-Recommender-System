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

# Number of results slider
num_results = st.slider("Number of books to return:", min_value=1, max_value=20, value=10, step=1)

top_k = st.slider("Number of initial search results to rerank (top_k):", min_value=10, max_value=50, value=30, step=10)

# Display the input
if user_input:
    st.write(f"You're looking for: {user_input}")
    
    # Get recommendations
    st.subheader("Recommendations")
    with st.spinner("Finding books for you..."):
        results = search_books(user_input, index, data_df, top_k=TOP_K)[['Title', 'Authors', 'Description']]

    with st.spinner("Reranking.."):
        results_reranked = rerank_books(user_input, results)[['Title', 'Authors', 'Description']].head(num_results)

    # Display results
    for idx, row in results_reranked.iterrows():
        st.markdown(f"### {idx + 1}. {row['Title']}")
        st.caption(f"*by {row['Authors']}*")
        st.write(row['Description'])
        st.divider()