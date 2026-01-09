# Semantic Neighbors Retrieval: Problem Analysis and Solutions

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
