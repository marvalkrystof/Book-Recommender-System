import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

def search_books(query, index, data_df, query_model = "./models/bge-large-en-v1.5", top_k=10):
    # 1. Embed query
    query_model = SentenceTransformer(query_model)

    q_emb = query_model.encode(
        [query],
        normalize_embeddings=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2. FAISS search
    scores, ids = index.search(q_emb, top_k)

    # 3. results DataFrame
    result_rows = []
    for score, idx in zip(scores[0], ids[0]):
        result_rows.append({
            "Book_ID": int(idx),
            "Title": data_df.iloc[idx]["Title"],
            "Authors": data_df.iloc[idx]["Authors"],
            "Description": data_df.iloc[idx]["Description"],
            "Topic": data_df.iloc[idx].get("Topic_Label", None),
            "Score": float(score)
        })
    return pd.DataFrame(result_rows)


def rerank_books(query, initial_results_df, reranker_model="./models/bge-reranker-large"):
    """
    Reranks search results using a cross-encoder model.
    
    Args:
        query: User's search query string
        initial_results_df: DataFrame with columns ['Title', 'Authors', 'Description']
        reranker_model: CrossEncoder model name
    
    Returns:
        DataFrame with reranked results and updated scores
    """
    reranker = CrossEncoder(reranker_model, device="cuda" if torch.cuda.is_available() else "cpu")

    # Create query-document pairs
    pairs = []
    for _, row in initial_results_df.iterrows():
        text = f"Title: {row['Title']} | Author: {row['Authors']} | Description: {row['Description']}"
        pairs.append([query, text])

    # Get reranking scores
    rerank_scores = reranker.predict(pairs)
    
    # Add scores and sort
    result_df = initial_results_df.copy()
    result_df['Rerank_Score'] = rerank_scores
    result_df = result_df.sort_values('Rerank_Score', ascending=False).reset_index(drop=True)
    
    return result_df
