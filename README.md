# Book-Recommender-System

A book recommendation system that uses machine learning to suggest books based on user preferences.

Data source: https://www.kaggle.com/datasets/elvinrustam/books-dataset

## Book Recommender — Data Preprocessing

This directory contains the preprocessing pipeline used to prepare book metadata,
clean descriptions.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/marvalkrystof/Book-Recommender-System.git
cd Book-Recommender-System
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Running the App

First download the required data and models.

### Download Data Files

Download the processed data, embeddings, and FAISS index:

```bash
python -c "import gdown; gdown.download_folder('https://drive.google.com/drive/folders/1L3tYlrzn2vSmIm0U0BDQAI1fQjG85tKV', output='data/', quiet=False)"
```

### Download Models

Download the models needed for ranking locally.

#### Sentence Transformer for initial ranking

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-large-en-v1.5', local_dir='./models/bge-large-en-v1.5')"
```

#### Reranker

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-large', local_dir='./models/bge-reranker-large')"
```

### Run the Application

To run the Streamlit web application:

```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`
