# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Semantic Change Analysis Toolkit** for analyzing how word meanings evolve over time using contextualized embeddings from Transformer models. The toolkit uses Word Sense Induction (WSI) to cluster word usages and visualize semantic shifts between time periods.

## Common Commands

### Setup
```bash
pip install uv
uv sync
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg  # For ingestion
```

### Run Tests
```bash
uv run pytest tests/ -v
```

### Data Ingestion (run once per corpus)
```bash
uv run python run_ingest.py
```
This creates SQLite databases (`data/ingested_1800.db`, `data/ingested_1900.db`) from raw text files in `data_gutenberg/`.

### Run the GUI
```bash
uv run streamlit run gui.py
```

### Command-line Analysis
```bash
# Single word analysis
uv run python main.py --word factory --model bert-base-uncased

# Batch analysis (pre-compute embeddings for all shared nouns)
uv run python -m src.semantic_change.embeddings_generation --model roberta-base --max-samples 200

# Rank Semantic Change (find words with largest shift)
uv run python src/rank_semantic_change.py --output output/ranking.csv
```

## Architecture

### Two-Stage Pipeline

1. **Ingestion Stage** (model-agnostic): Raw text corpora are tokenized and lemmatized using spaCy, then stored in SQLite databases. This only needs to run once per corpus.

2. **Analysis Stage** (model-specific): The system queries sentences from SQLite, generates embeddings on-the-fly using the selected Transformer model, performs clustering (WSI), and creates visualizations.

### Core Modules (`semantic_change/`)

- **`corpus.py`**: `Corpus` and `CorpusManager` classes for SQLite-backed corpus access. Provides `query_samples()` for retrieving sentences containing a target word.

- **`embedding.py`**: `BertEmbedder` class using spacy-transformers for alignment between linguistic tokens and Transformer subwords. Key methods:
  - `batch_extract()`: Extract embeddings for specific word spans
  - `get_embeddings()`: Legacy convenience method
  - `get_nearest_neighbors()`: MLM head projection to find semantic neighbors

- **`ingestor.py`**: `Ingestor` class that processes raw `.txt` files into SQLite with three tables: `files`, `sentences`, `tokens`. Uses spaCy for tokenization/lemmatization.

- **`wsi.py`**: `WordSenseInductor` facade supporting KMeans, Spectral, Agglomerative, and HDBSCAN clustering algorithms.

- **`visualization.py`**: `Visualizer` class for Plotly-based interactive plots. Supports PCA/t-SNE/UMAP dimensionality reduction. Creates scatter plots and k-NN graph visualizations.

- **`vector_store.py`**: ChromaDB wrapper for caching pre-computed embeddings. Key methods:
  - `store_embeddings()`: Cache embeddings with metadata
  - `query_embeddings()`: Retrieve cached embeddings
  - `delete_model_embeddings()`: Delete all embeddings for a model/project

- **`config_manager.py`**: Configuration management using dataclass pattern. Provides:
  - `AppConfig`: Dataclass with all configuration fields and defaults
  - `load()` / `save()`: Persist configuration to JSON
  - `get_db_paths()` / `check_databases_exist()`: Database path utilities
  - Legacy wrapper functions for backward compatibility

- **`services.py`**: Business logic services (MVC pattern). Contains:
  - `StatsService`: Corpus and embedding statistics retrieval
  - `ClusterService`: Cluster operations (save for drill-down analysis)
  - `CorpusStats` / `EmbeddingStats`: Data classes for statistics

### Utilities (`utils/`)

- **`dependencies.py`**: Dependency checking for spaCy transformer models

### Entry Points

- **`gui.py`**: Wrapper for `src/gui_app.py`. Streamlit interface with pages for Analysis Dashboard, Data Ingestion, Embeddings Config, and Corpus Reports. Configuration stored in `config.json`.

- **`main.py`**: `run_single_analysis()` function for CLI-based single-word analysis.

- **`src/semantic_change/embeddings_generation.py`**: Pre-computes and caches embeddings for all shared nouns between corpora.

- **`src/rank_semantic_change.py`**: Calculates semantic shift (cosine distance) for shared words.

- **`run_ingest.py`**: Standalone ingestion script.

### Data Flow

```
Raw .txt files (data_gutenberg/1800/, data_gutenberg/1900/)
    |
    v  [Ingestor + spaCy]
SQLite DBs (data/ingested_*.db)
    |
    v  [Corpus.query_samples()]
Sentences with target word
    |
    v  [BertEmbedder + spacy-transformers]
Contextual embeddings
    |
    v  [WordSenseInductor]
Cluster labels
    |
    v  [Visualizer]
Interactive HTML plots
```

## Key Configuration Parameters

- `model_name`: HuggingFace model ID (e.g., `bert-base-uncased`, `answerdotai/ModernBERT-base`)
- `n_samples`: Number of sentences to sample per corpus
- `min_cluster_size`: HDBSCAN minimum cluster size
- `n_clusters`: For KMeans/Spectral/Agglomerative
- `wsi_algorithm`: `hdbscan`, `kmeans`, `spectral`, or `agglomerative`
- `context_window`: 0 for sentence-only, >0 for surrounding sentences

## Project Structure

```
├── gui.py                  # Streamlit entry point
├── main.py                 # CLI entry point
├── config.json             # Runtime configuration
├── pyproject.toml          # Dependencies and pytest config
├── src/
│   ├── gui_app.py          # Streamlit UI (view layer)
│   ├── main.py             # CLI analysis logic
│   ├── semantic_change/
│   │   ├── config_manager.py   # Configuration dataclass
│   │   ├── services.py         # Business logic services
│   │   ├── corpus.py           # SQLite corpus access
│   │   ├── embedding.py        # Transformer embeddings
│   │   ├── vector_store.py     # ChromaDB cache
│   │   ├── wsi.py              # Clustering algorithms
│   │   └── visualization.py    # Plotly visualizations
│   └── utils/
│       └── dependencies.py     # Dependency checking
├── tests/
│   ├── conftest.py             # Pytest configuration
│   ├── test_config_manager.py  # Config tests
│   ├── test_services.py        # Service layer tests
│   ├── test_dependencies.py    # Dependency check tests
│   └── test_vector_store.py    # VectorStore tests
└── data/
    ├── corpus_t1.db        # Ingested corpus (period 1)
    ├── corpus_t2.db        # Ingested corpus (period 2)
    └── chroma_db/          # Cached embeddings
```

## Code Style

- PEP 8 with type hints
- Docstrings for functions and classes
- MVC pattern: GUI handles rendering, services handle business logic
