# Completed Tasks

## Core Infrastructure
- [x] **Repo Setup**: Created initial git repository and directory structure.
- [x] **Ingestion Pipeline**: Implemented `Ingestor` class using spaCy to process raw text into SQLite.
    - [x] GPU support enabled for spaCy.
    - [x] Progress bar for ingestion steps.
- [x] **Batch Processing**: Implemented `run_batch_analysis.py` to generate embeddings for whole vocabularies.
    - [x] Refactored to support independent `t1`/`t2` corpora.
    - [x] Removed NLTK dictionary dependencies to support multi-language/historical text.
    - [x] Integrated `ChromaDB` for vector storage.

## Analysis & Visualization
- [x] **Word Sense Induction**: Implemented clustering (HDBSCAN/KMeans) of BERT embeddings.
- [x] **Visualization**: Interactive Plotly charts for:
    - Time Period distribution.
    - Sense Clusters.
    - Combined Sense x Time view.
- [x] **Semantic Neighbors**:
    - [x] Replaced unstable MLM projection with direct Vector Store retrieval.
    - [x] Implemented iterative querying to find diverse, unique lemmas as neighbors.
    - [x] Added filename tracking to trace sentences back to source files.

## GUI (Streamlit)
- [x] **Navigation**: Structured into Dashboard, Ingestion, Embeddings, and Reports tabs.
- [x] **Multi-Model Support**:
    - [x] Dropdown to select available embedding sets from the DB.
    - [x] "Create New" flow to generate embeddings for arbitrary HuggingFace models.
- [x] **Data Management**:
    - [x] "Wipe Database" options in the Ingestion tab.
    - [x] Ability to delete specific embedding collections.

## Documentation
- [x] **Architecture**: Created `architecture.md` describing the system design.
- [x] **Code**: Refactored naming to generic `t1`/`t2` convention.
