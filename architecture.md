# System Architecture

## Overview

The SILD (Semantic Change Analysis) system allows users to analyze how the meaning of words shifts between two distinct time periods (T1 and T2). It processes raw text corpora, generates contextualized word embeddings using Large Language Models (LLMs), and clusters these embeddings to identify and visualize distinct word senses.

## Data Flow Pipeline

```ascii
+-------------------+       +-------------------+
| Raw Text Corpora  |       |   Spacy Model     |
| (data_source/t1)  |       | (en_core_web_lg)  |
| (data_source/t2)  |       |                   |
+---------+---------+       +---------+---------+
          |                           |
          v                           v
+---------------------------------------------------+
|                 Ingestion (Stage 1)               |
|           (src/semantic_change/ingestor.py)       |
|                                                   |
| 1. Tokenization & Sentence Splitting              |
| 2. Lemmatization & POS Tagging                    |
| 3. Storage in Structured SQLite DB                |
+-------------------------+-------------------------+
                          |
                          v
+---------------------------------------------------+
|                 SQLite Databases                  |
|          (data/corpus_t1.db, corpus_t2.db)        |
|                                                   |
| Tables:                                           |
| - files (id, filepath)                            |
| - sentences (id, text, file_offset)               |
| - tokens (lemma, pos, start_char, sentence_id)    |
+-------------------------+-------------------------+
                          |
                          v
+---------------------------------------------------+
|              Batch Analysis (Stage 2)             |
|             (src/run_batch_analysis.py)           |
|                                                   |
| 1. Extract Frequent Vocabulary (per corpus)       |
| 2. Retrieve Sentences for each word               |
| 3. Generate Embeddings (BERT/ModernBERT)          |
| 4. Store Vector + Metadata                        |
+-------------------------+-------------------------+
                          |
                          v
+---------------------------------------------------+
|             Vector Store (ChromaDB)               |
|               (data/chroma_db)                    |
|                                                   |
| Collections:                                      |
| - embeddings_t1_{model_name}                      |
| - embeddings_t2_{model_name}                      |
|                                                   |
| Stored Data:                                      |
| - Vector (768d float array)                       |
| - Metadata: {lemma, sentence_id, offsets}         |
+-------------------------+-------------------------+
                          |
                          v
+---------------------------------------------------+
|             Analysis & Visualization              |
|                  (src/main.py)                    |
|                                                   |
| 1. Query Embeddings for Target Word               |
| 2. Dimensionality Reduction (PCA/UMAP)            |
| 3. Clustering (HDBSCAN/KMeans) -> "Senses"        |
| 4. Neighbor Retrieval (via Vector Similarity)     |
| 5. Interactive Plotting (Plotly)                  |
+---------------------------------------------------+
```

## Component Details

### 1. Ingestion (`src/semantic_change/ingestor.py`)
- **Purpose**: Converts raw unstructured text into a queryable structure.
- **Key Feature**: Uses `spacy` (GPU-accelerated if available) for robust linguistic annotation.
- **Output**: SQLite databases that allow fast retrieval of all sentences containing a specific lemma.

### 2. Batch Embedding (`src/run_batch_analysis.py`)
- **Purpose**: Pre-computes context-aware embeddings for the entire relevant vocabulary.
- **Strategy**: 
    - Identifies all Nouns/Verbs/Adjectives with frequency > `min_freq` in each corpus independently.
    - Samples up to `max_samples` sentences for each word.
    - Computes embeddings using the specified HuggingFace model.
- **Storage**: Saves embeddings to ChromaDB, enabling semantic search (Nearest Neighbors) later.

### 3. Analysis Core (`src/main.py`)
- **Purpose**: Performs the actual Word Sense Induction (WSI).
- **Process**:
    - Fetches embeddings for the target word from ChromaDB.
    - Clusters them to find distinct senses (e.g., "bank" -> [river bank, financial bank]).
    - Visualizes the distribution over time.
    - **Semantic Neighbors**: Finds the closest *other* words in the vector space to describe each sense cluster. It uses iterative querying of ChromaDB to ensure a diverse set of real-word neighbors.

## Directory Structure

```
project_root/
├── data/                   # Generated artifacts (DBs, Embeddings)
│   ├── corpus_t1.db        # SQLite DB for Period 1
│   ├── corpus_t2.db        # SQLite DB for Period 2
│   └── chroma_db/          # Vector Store
├── data_source/            # Input Raw Text
│   ├── t1/                 # Text files for Period 1
│   └── t2/                 # Text files for Period 2
├── output/                 # Visualization HTMLs
└── src/                    # Source Code
    ├── gui_app.py          # Streamlit Dashboard
    ├── main.py             # Analysis Logic
    ├── run_batch_analysis.py # Embedding Generation
    ├── run_ingest.py       # Corpus Ingestion
    └── semantic_change/    # Core Modules
        ├── corpus.py       # DB Interface
        ├── embedding.py    # LLM Wrapper
        ├── ingestor.py     # Spacy Processing
        ├── vector_store.py # ChromaDB Wrapper
        └── visualization.py # Plotly Charts
```
