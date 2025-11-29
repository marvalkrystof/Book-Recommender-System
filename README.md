# Book-Recommender-System

Data source: https://www.kaggle.com/datasets/elvinrustam/books-dataset

# Book Recommender — Data Preprocessing

This directory contains the preprocessing pipeline used to prepare book metadata,
clean descriptions, generate embeddings, and build the FAISS index.

## Contents
- `data_preprocess_2_3_clean.ipynb` — cleaned Jupyter notebook ready for GitHub.
- Embedding generation code using `BAAI/bge-large-en-v1.5`
- FAISS index construction for similarity search

## Pipeline Overview
1. **Combine metadata into a final text field**  
   Description + topic labels + keywords are merged for richer embeddings.

2. **Generate text embeddings**  
   Using a SentenceTransformer model on CUDA if available.

3. **Build a FAISS index**  
   We use `IndexFlatIP` for cosine similarity over normalized vectors for fast retrieval.

## How to Run
1. Install dependencies:
   ```bash
   pip install sentence-transformers faiss-cpu pandas numpy
   ```
2. Open the cleaned notebook:
   ```bash
   jupyter notebook data_preprocess_2_3_clean.ipynb
   ```

## Notes
- Outputs and execution counts were cleared for GitHub readability.
- No large data files are included in the notebook for repo cleanliness.
