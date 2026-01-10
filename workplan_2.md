# Work Plan Phase 2: Variable Context & Robust Alignment
**Date:** 2026-01-05
**Status:** Ingestor Refactored, Environment Prepared.

## 1. Executive Summary of Architectural Decisions
Today we identified a fundamental misalignment between the linguistic layer (spaCy) and the representation layer (BERT/ModernBERT). To solve this and enable future features, we have committed to the following:

### What we are doing (The "Why"):
*   **Hybrid Pointer-Content Storage:** We are storing sentence text in SQLite for performance, but adding `file_offset_start` pointers to the raw text files. 
    *   *Reason:* This allows us to "break out" of the sentence boundary and fetch arbitrary context (up to 8k tokens for ModernBERT) without re-ingesting the whole database.
*   **`spacy-transformers` Integration:** We are moving away from manual character-overlap math to the industry-standard alignment provided by `spacy-transformers`.
    *   *Reason:* It handles the complex mapping between linguistic tokens and sub-word units (like RoPE-based ModernBERT tokens) natively and robustly.
*   **Model-Based Namespacing:** Embeddings are now stored in ChromaDB collections named by the model (e.g., `embeddings_1800_bert_base_uncased`).
    *   *Reason:* Embeddings from different models are mathematically incompatible and must never be mixed or averaged together.
*   **Configurable Linguistic Layer:** The spaCy model is now a variable in the config/GUI.
    *   *Reason:* To support multi-language analysis (e.g., German or French corpora) which require different lemmatizers.

### What we decided NOT to do:
*   **Fixed Chunking (ElasticSearch Style):** We rejected pre-defined windowing because it is optimized for retrieval, whereas our project requires precise word-level analysis with flexible boundaries.
*   **Inference-at-Ingest:** we are not running transformer models during the ingestion phase. 
    *   *Reason:* It would make ingestion 100x slower and lock the database to a single specific model. We keep ingestion linguistic-only (lemmas/POS) and run embeddings "just-in-time".

---

## 2. Tasks for Tomorrow

### Task 1: Overhaul `semantic_change/embedding.py`
*   **Implement `TransformerAligner`:** Create a class that uses `spacy-transformers` to wrap the HuggingFace model.
*   **Logic:** Feed the text (sentence or extended context) into the pipeline, and use the `.ext._.trf_data` alignment map to extract the exact vectors for the target word.
*   **Fallback:** Maintain a robust fallback for models not yet supported by the spaCy-transformers wrapper.

### Task 2: Implement "Dynamic Context" in `main.py`
*   **`ContextRetriever` Helper:** Write logic that checks the user-requested `context_size`. 
    *   If `size == "sentence"`, use the DB text.
    *   If `size > sentence`, use `file_offset_start` to read the raw file around the word.
*   **API Update:** Update `run_single_analysis` to accept a `context_window` parameter.

### Task 3: GUI Refinement
*   **Linguistic Settings:** Add a selection for "Linguistic Model" (spaCy) in the Data Ingestion tab.
*   **Analysis Settings:** Add a slider for "Context Window" (number of tokens/characters around the target word).
*   **Status Indicators:** Show which model's embeddings are currently loaded from ChromaDB.

### Task 4: Full System Verification
*   **Re-Ingest:** Run the updated Ingestor on both corpora to populate the new `file_offset_start` columns.
*   **BERT vs. ModernBERT:** Run "well" (NOUN) through both models. 
    *   *BERT Goal:* Verify clean neighbors (spring, water).
    *   *ModernBERT Goal:* Investigate if `spacy-transformers` alignment solves the "suffix-neighbor" issue seen today.

### Task 5: Extended Reporting (Item 11.3)
*   Finalize the `rank_semantic_change.py` script to generate the CSV report using the new namespaced collections.
