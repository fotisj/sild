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



## The Problem

The current implementation retrieves semantic neighbors by projecting a cluster centroid through the **MLM (Masked Language Model) head** of the transformer. This projects the embedding into vocabulary space and returns the highest-probability tokens.

### Why this fails with ModernBERT (and other BPE tokenizers)

1. **Tokenizer vocabulary structure differs**:
   - **BERT (WordPiece)**: Many complete words exist as single tokens (e.g., "electricity" = 1 token)
   - **ModernBERT/RoBERTa (BPE)**: Vocabulary is dominated by subword units (e.g., "electricity" = `["Ġelect", "ricity"]`)

2. **Top-k predictions are subword fragments**:
   - The MLM head returns token IDs from the vocabulary
   - For BPE tokenizers, the most frequent/probable tokens are short subword units
   - Results: `['whe', 'com', 'cl', 'cont', 'app']` instead of meaningful words

3. **Filtering cannot fix this**:
   - Longer words (multi-token) **don't exist** in vocabulary space as single entries
   - Filtering out fragments doesn't reveal hidden good results - those results simply aren't representable
   - We tried: subword marker detection, NLTK dictionary validation, spaCy vectors - all insufficient

### Root Cause

The MLM head projection asks: *"What single token is closest to this vector?"*

For BPE tokenizers, the answer is almost always a common subword fragment, not a complete word.

---

## Solution 1: ChromaDB Word Embeddings

### Approach

During batch processing, store contextual embeddings for words in the corpus. At query time, find words whose embeddings are closest (cosine similarity) to the cluster centroid.

### Implementation Outline

1. **During batch analysis** (`run_batch_analysis.py`):
   - Already stores embeddings in ChromaDB with metadata (lemma, sentence_id, etc.)
   - Could aggregate to per-word centroids or keep per-occurrence embeddings

2. **At query time**:
   - Compute cluster centroid
   - Query ChromaDB for nearest neighbors by vector similarity
   - Return words (not tokens) with highest similarity

### Pros

| Advantage | Detail |
|-----------|--------|
| Fast retrieval | ChromaDB is optimized for vector similarity search |
| Works with any tokenizer | Operates in embedding space, not vocabulary space |
| Returns real words | By design - we store words, not subword tokens |
| Existing infrastructure | `VectorStore` class and batch analysis already exist |
| Straightforward | Conceptually simple vector similarity search |

### Cons

| Disadvantage | Detail |
|--------------|--------|
| Requires pre-computation | Must run batch analysis before this works |
| Storage overhead | Scales with corpus size × vocabulary |
| Limited to corpus vocabulary | Cannot surface words not in the corpus |
| Sense disambiguation challenge | Same word has different embeddings in different contexts |
| **Critical limitation** | If vocabulary is restricted to words in BOTH corpora (current behavior), disappearing word senses won't have meaningful neighbors |

---

## Solution 2: Contextual MLM Aggregation

### Approach

Instead of projecting an abstract centroid, use MLM prediction in actual sentence contexts and aggregate results:

1. Find n sentences whose focus word embeddings are nearest to the cluster centroid
2. For each sentence, mask the focus word
3. Run MLM prediction to get top-k replacement words
4. Aggregate predictions across all n sentences
5. Return the m most frequently predicted replacements

### Implementation Outline

1. **Get sentences near centroid**:
   - Use existing embeddings to find n closest occurrences
   - Retrieve original sentences from SQLite

2. **Contextual MLM prediction**:
   - Tokenize sentence, identify focus word token(s)
   - Mask the focus word (all tokens if multi-token)
   - Get top-k predictions for the first (word-start) position
   - Filter to complete words only

3. **Aggregation**:
   - Count prediction frequencies across all n sentences
   - Return top m most frequent predictions

### Handling Multi-Token Focus Words

When the focus word spans multiple tokens (e.g., "electricity" → `[Ġelect, ricity]`):

- Mask ALL tokens belonging to the word
- Take predictions from the **first position only** (word-start position)
- Predictions will be word-start tokens, many of which are complete words
- Filter to keep only single-token complete words

This works because semantically similar alternatives to multi-token words are often common single-token words (e.g., "power" instead of "electricity").

### Pros

| Advantage | Detail |
|-----------|--------|
| Context-sensitive | Predictions grounded in actual sentence contexts |
| Natural noise filtering | Fragments won't be consistently predicted across diverse contexts |
| No pre-computation | Works on-the-fly without batch processing |
| Not corpus-limited | Can surface any word in the model's vocabulary |
| Sense-appropriate | Sentences near centroid represent that specific sense |
| Linguistically principled | Asks "what word fits here?" not "what token is close?" |

### Cons

| Disadvantage | Detail |
|--------------|--------|
| Slower | Requires n MLM inference passes per cluster |
| Still single-token predictions | Must filter to complete words |
| Parameter tuning | Need to choose n (sentences) and k (predictions per sentence) |
| Multi-token handling | First-position predictions may miss some alternatives |
| Context sensitivity | Unusual sentences could skew results |

### Suggested Default Parameters

- `n = 10` sentences near centroid
- `k = 6` predictions per sentence
- `m = 10` final neighbors to display
- These can be exposed in the GUI for experimentation

---

## Recommendation

**Solution 2 (Contextual MLM Aggregation)** is preferred because:

1. **Aggregation solves the fragment problem**: A fragment like "whe" might occasionally be predicted, but real semantic alternatives will dominate across multiple contexts.

2. **Handles disappearing senses**: Not limited to words appearing in both corpora.

3. **Semantically grounded**: Predictions reflect actual usage patterns in context.

4. **No infrastructure changes**: Doesn't require modifying the batch analysis or storage.

5. **Aligns with research goal**: The task is understanding word meaning in context - this approach directly uses contextual information.

---

## Implementation Priority

1. Implement Solution 2 (Contextual MLM Aggregation)
2. Add GUI controls for n, k, m parameters
3. Keep existing MLM head projection as fallback/comparison
4. Consider Solution 1 as future enhancement for speed-critical use cases

---

## Technical Notes

### Tokenizer Style Detection

ModernBERT shows both BERT and RoBERTa markers in vocabulary:
```
[DEBUG] Tokenizer style detection: BERT=True, RoBERTa=True, SentencePiece=False
```

This hybrid tokenizer requires careful handling:
- `##` prefix = BERT-style continuation token
- `Ġ` prefix = RoBERTa-style word-start token
- Tokens with neither = ambiguous (skip these)

### Current Code Location

The neighbor retrieval logic is in:
- `src/semantic_change/embedding.py` → `BertEmbedder.get_nearest_neighbors()`

Called from:
- `src/main.py` → `run_single_analysis()` (lines 341-364)


=====================================================

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
