import os
import sys
# Suppress HuggingFace xet storage warning (falls back to HTTP which works fine)
os.environ["HF_HUB_DISABLE_XET_WARNING"] = "1"

import torch
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from typing import List, Tuple, Dict, Any, Optional
import warnings
import logging

# Suppress huggingface_hub xet warnings
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)

# Available pooling strategies for subword token aggregation
POOLING_STRATEGIES = {
    "mean": "Mean of all subword tokens (default, may be influenced by morphology)",
    "first": "First subword token only (often carries most semantic content)",
    "lemma_aligned": "Only pool subwords matching the lemma's tokenization length",
    "weighted": "Position-weighted pooling (earlier subwords get higher weight)",
    "lemma_replacement": "Replace target word with lemma before embedding (TokLem, Laicher et al. 2021)",
}


def detect_optimal_dtype() -> Optional[torch.dtype]:
    """
    Detects the optimal mixed precision dtype based on GPU capabilities.

    Returns:
        torch.bfloat16 for datacenter GPUs (A100, L40, H100, etc.)
        torch.float16 for consumer GPUs (RTX 30xx, 40xx, etc.)
        None for CPU or unsupported GPUs
    """
    if not torch.cuda.is_available():
        return None

    device_name = torch.cuda.get_device_name(0).lower()
    capability = torch.cuda.get_device_capability(0)
    major, minor = capability

    # Check for bf16 support (compute capability 8.0+ has good bf16)
    # Datacenter GPUs: A100 (8.0), L40 (8.9), H100 (9.0) - use bf16
    # Consumer GPUs: RTX 3090 (8.6), RTX 4090 (8.9) - fp16 is faster

    # Datacenter GPUs that benefit from bf16
    datacenter_keywords = ['a100', 'a10', 'l40', 'h100', 'h200', 'v100', 'a30', 'a40']
    is_datacenter = any(kw in device_name for kw in datacenter_keywords)

    if is_datacenter and major >= 8:
        return torch.bfloat16
    elif major >= 7:  # Volta (7.0) and newer support fp16 well
        return torch.float16
    else:
        return None


def detect_optimal_batch_size() -> int:
    """
    Detects an optimal batch size based on available GPU VRAM.
    Memory cleanup between batches allows for larger batch sizes.

    Returns:
        Recommended batch size for embedding extraction.
        - 256 for GPUs with 40GB+ VRAM (A100, L40, etc.)
        - 128 for GPUs with 24GB+ VRAM (RTX 3090/4090, A5000, etc.)
        - 64 for GPUs with 12GB+ VRAM (RTX 3080, etc.)
        - 32 for GPUs with less VRAM or CPU
    """
    if not torch.cuda.is_available():
        return 32

    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        if vram_gb >= 40:
            return 256
        elif vram_gb >= 24:
            return 128
        elif vram_gb >= 12:
            return 64
        else:
            return 32
    except Exception:
        return 64

class Embedder:
    """Base class for embedding generation."""
    def get_embeddings(self, samples: List[Dict[str, str]]) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError

class BertEmbedder(Embedder):
    """
    Generates contextual embeddings using a Transformer model and provides MLM-based decoding.
    Allows configuration of which layers to use and how to combine them.
    Uses direct HuggingFace transformers with offset_mapping for character-to-subword alignment.

    The MLM model and filter model are loaded lazily on first call to get_nearest_neighbors()
    to reduce memory usage and startup time for batch embedding extraction.
    """
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = None,
                 layers: List[int] = None, layer_op: str = 'mean', lang: str = 'en',
                 filter_model: str = None, pooling_strategy: str = 'mean'):
        """
        Args:
            model_name: HuggingFace model name.
            device: 'cpu' or 'cuda'.
            layers: List of layer indices to use (e.g., [-1, -2, -3, -4]). Default is [-1].
                    Supports negative indexing: -1 is last layer, -2 is second-to-last, etc.
                    Layer 0 is the input embeddings, layers 1-12 are transformer layers (for BERT-base).
            layer_op: How to combine layers: 'mean', 'sum', or 'concat'. Default is 'mean'.
                      'concat' will multiply output dimension by len(layers).
            lang: Language code for the spaCy filter model.
            filter_model: spaCy model for neighbor filtering (lemma/pos). If None, uses a default based on lang.
            pooling_strategy: How to pool subword tokens. Options:
                - 'mean': Mean of all subword tokens (default)
                - 'first': First subword token only
                - 'lemma_aligned': Only pool subwords matching lemma's tokenization length
                - 'weighted': Position-weighted pooling (earlier tokens weighted higher)
                - 'lemma_replacement': Replace target with lemma in text before embedding (TokLem)
        """
        if pooling_strategy not in POOLING_STRATEGIES:
            raise ValueError(f"Unknown pooling_strategy '{pooling_strategy}'. "
                           f"Valid options: {list(POOLING_STRATEGIES.keys())}")

        self.model_name = model_name
        self.lang = lang
        self.filter_model_name = filter_model
        self.pooling_strategy = pooling_strategy
        self.device_name = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device_name)

        # Detect optimal mixed precision dtype for this GPU
        self.mixed_precision_dtype = detect_optimal_dtype()

        import sys
        print(f"Loading model '{model_name}' on {self.device_name}...")
        print(f"  Pooling strategy: {pooling_strategy}")
        if self.mixed_precision_dtype:
            dtype_name = 'bfloat16' if self.mixed_precision_dtype == torch.bfloat16 else 'float16'
            print(f"  Mixed precision enabled: {dtype_name}")
        sys.stdout.flush()

        # Load transformer model directly with HuggingFace (enables multi-layer access)
        try:
            print(f"  Loading transformer model with output_hidden_states=True...")
            sys.stdout.flush()
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)
            self.model.eval()
            print(f"  Model loaded on {self.device_name}.")
            sys.stdout.flush()

            print(f"  Loading tokenizer...")
            sys.stdout.flush()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            print(f"  Tokenizer loaded.")
            sys.stdout.flush()

        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            sys.stdout.flush()
            raise

        # Lazy-loaded components (initialized on first use in get_nearest_neighbors)
        self._mlm_model = None  # Will be loaded on first neighbor query
        self._filter_nlp = None  # Will be loaded on first neighbor query
        self._mlm_load_failed = False  # Track if MLM loading already failed

        self.layers = layers if layers is not None else [-1]
        self.layer_op = layer_op.lower()

        # Validate layer_op
        if self.layer_op not in ('mean', 'median', 'sum', 'concat'):
            raise ValueError(f"Unknown layer_op '{layer_op}'. Valid options: 'mean', 'median', 'sum', 'concat'")

        print(f"BertEmbedder initialization complete for '{model_name}'.")
        print(f"  Layers: {self.layers}, Layer operation: {self.layer_op}")
        print(f"  (MLM head and filter model will be loaded on first neighbor query)")
        sys.stdout.flush()

    def _ensure_mlm_model(self):
        """Lazily loads the MLM model on first access."""
        if self._mlm_model is not None or self._mlm_load_failed:
            return self._mlm_model

        import sys
        import logging

        # Suppress expected warning about unused pooler weights
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        print(f"  Loading MLM head (for semantic neighbors)...")
        sys.stdout.flush()

        try:
            self._mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
            self._mlm_model.eval()
            print(f"  MLM head loaded.")
            sys.stdout.flush()
        except Exception as e:
            print(f"  Warning: Could not load MLM head for '{self.model_name}': {e}")
            print("  Semantic neighbors feature will be disabled for this model.")
            sys.stdout.flush()
            self._mlm_load_failed = True

        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
        return self._mlm_model

    def _ensure_filter_nlp(self):
        """Lazily loads the spaCy filter model on first access."""
        if self._filter_nlp is not None:
            return self._filter_nlp

        import sys

        # Determine which model to load
        if self.filter_model_name:
            filter_model_to_load = self.filter_model_name
        else:
            # Default models by language
            default_models = {
                'en': 'en_core_web_lg',
                'de': 'de_core_news_lg',
                'fr': 'fr_core_news_lg',
                'es': 'es_core_news_lg',
                'it': 'it_core_news_lg',
                'nl': 'nl_core_news_lg',
                'pt': 'pt_core_news_lg',
            }
            filter_model_to_load = default_models.get(self.lang, f'{self.lang}_core_news_lg')

        try:
            print(f"  Loading filter model '{filter_model_to_load}'...")
            sys.stdout.flush()
            self._filter_nlp = spacy.load(filter_model_to_load, disable=["parser", "ner"])
            print(f"  Filter model loaded.")
            sys.stdout.flush()
        except OSError:
            print(f"  Warning: spaCy model '{filter_model_to_load}' not found. Neighbor filtering might be limited.")
            sys.stdout.flush()

        return self._filter_nlp

    @property
    def mlm_model(self):
        """Property for backward compatibility - lazily loads MLM model."""
        return self._ensure_mlm_model()

    @property
    def filter_nlp(self):
        """Property for backward compatibility - lazily loads filter model."""
        return self._ensure_filter_nlp()

    def _get_subword_indices_for_span(
        self, offset_mapping: List[Tuple[int, int]], start_char: int, end_char: int
    ) -> List[int]:
        """
        Find subword token indices that overlap with a character span.

        Args:
            offset_mapping: List of (start_char, end_char) tuples from tokenizer
            start_char: Start character position of target word
            end_char: End character position of target word

        Returns:
            List of token indices that overlap with the character span
        """
        indices = []
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            # Skip special tokens (offset_mapping shows (0, 0) for special tokens)
            if tok_start == 0 and tok_end == 0:
                continue
            # Check if token overlaps with target span
            if tok_start < end_char and tok_end > start_char:
                indices.append(idx)
        return indices

    def _combine_layers(
        self, hidden_states: Tuple[torch.Tensor, ...], token_indices: List[int]
    ) -> torch.Tensor:
        """
        Extract and combine hidden states from specified layers for given token indices.

        Args:
            hidden_states: Tuple of tensors from model output, each (batch, seq_len, hidden_dim).
                          hidden_states[0] is input embeddings, [1] to [12] are transformer layers (BERT-base).
            token_indices: List of token indices to extract embeddings for

        Returns:
            Tensor of shape (num_tokens, output_dim) where output_dim depends on layer_op:
            - 'mean' or 'sum': output_dim = hidden_dim
            - 'concat': output_dim = hidden_dim * len(layers)
        """
        num_total_layers = len(hidden_states)

        # Convert negative indices to positive (e.g., -1 -> 12 for 13-layer BERT)
        layer_indices = [l if l >= 0 else num_total_layers + l for l in self.layers]

        # Validate layer indices
        for idx in layer_indices:
            if idx < 0 or idx >= num_total_layers:
                raise ValueError(
                    f"Layer index {idx} out of range. Model has {num_total_layers} layers (0 to {num_total_layers-1})."
                )

        # Extract embeddings for specified tokens from each selected layer
        # Each hidden_states[layer] is (batch=1, seq_len, hidden_dim)
        # We want (num_tokens, num_layers, hidden_dim)
        layer_embeddings = []
        for layer_idx in layer_indices:
            # Extract tokens: (num_tokens, hidden_dim)
            token_embeds = hidden_states[layer_idx][0, token_indices, :]
            layer_embeddings.append(token_embeds)

        # Stack layers: (num_layers, num_tokens, hidden_dim)
        stacked = torch.stack(layer_embeddings, dim=0)
        # Transpose to (num_tokens, num_layers, hidden_dim)
        stacked = stacked.permute(1, 0, 2)

        # Combine according to layer_op
        if self.layer_op == 'mean':
            # (num_tokens, hidden_dim)
            return stacked.mean(dim=1)
        elif self.layer_op == 'median':
            # (num_tokens, hidden_dim)
            return stacked.median(dim=1).values
        elif self.layer_op == 'sum':
            # (num_tokens, hidden_dim)
            return stacked.sum(dim=1)
        elif self.layer_op == 'concat':
            # (num_tokens, num_layers * hidden_dim)
            return stacked.flatten(start_dim=1)
        else:
            raise ValueError(f"Unknown layer_op: {self.layer_op}")

    def _pool_subwords(self, sub_vectors: torch.Tensor, lemma: str = None) -> torch.Tensor:
        """
        Pool subword token vectors according to the configured pooling strategy.

        Args:
            sub_vectors: Tensor of shape (num_subtokens, dim) with subword embeddings
            lemma: The lemma of the target word (needed for lemma_aligned strategy)

        Returns:
            Single pooled vector of shape (dim,)
        """
        if len(sub_vectors) == 1:
            return sub_vectors[0]

        if self.pooling_strategy == "mean":
            return torch.mean(sub_vectors, dim=0)

        elif self.pooling_strategy == "first":
            return sub_vectors[0]

        elif self.pooling_strategy == "weighted":
            # Position-weighted: earlier subwords get higher weight
            n = len(sub_vectors)
            weights = 1.0 / torch.arange(1, n + 1, device=sub_vectors.device, dtype=sub_vectors.dtype)
            weights = weights / weights.sum()
            return (sub_vectors * weights.unsqueeze(1)).sum(dim=0)

        elif self.pooling_strategy == "lemma_aligned":
            # Only pool subwords up to the lemma's tokenization length
            if lemma:
                lemma_tokens = self.tokenizer.tokenize(lemma)
                n_lemma = len(lemma_tokens)
                if n_lemma > 0 and n_lemma < len(sub_vectors):
                    return torch.mean(sub_vectors[:n_lemma], dim=0)
            # Fallback to mean if lemma not provided or longer than actual tokens
            return torch.mean(sub_vectors, dim=0)

        else:
            # lemma_replacement is handled at preprocessing stage, fallback to mean
            return torch.mean(sub_vectors, dim=0)

    def _preprocess_for_lemma_replacement(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Preprocess items for lemma_replacement strategy (TokLem).
        Replaces each target word in the text with its lemma.

        For each item with multiple targets, creates separate items
        (one per target) with the word replaced by its lemma.

        Args:
            items: List of dicts with 'text', 'id', 'targets' [(lemma, pos, start, end), ...]

        Returns:
            Modified items with lemma-replaced texts and adjusted offsets
        """
        modified_items = []

        for item in items:
            text = item['text']
            sent_id = item['id']

            for lemma, pos, start_char, end_char in item['targets']:
                original_word = text[start_char:end_char]

                # Use lemma as-is to preserve case (important for cased models)
                # SpaCy lemmas already have correct case (e.g., German nouns capitalized)
                replacement = lemma

                # Create modified text with lemma replacing the target word
                modified_text = text[:start_char] + replacement + text[end_char:]

                # New end position after replacement
                new_end_char = start_char + len(replacement)

                modified_items.append({
                    'id': sent_id,
                    'text': modified_text,
                    'targets': [(lemma, pos, start_char, new_end_char)],
                    'original_token': original_word,  # Keep for metadata
                })

        return modified_items

    def batch_extract(self, items: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Extracts embeddings for specific words using direct HuggingFace transformers.
        Uses offset_mapping for character-to-subword alignment.
        Supports true multi-layer extraction via output_hidden_states.
        Uses mixed precision (bf16/fp16) when available for ~2x speedup.

        The pooling strategy (configured in __init__) determines how subword tokens are aggregated:
        - 'mean': Mean of all subword tokens
        - 'first': First subword token only
        - 'lemma_aligned': Only pool subwords matching lemma's tokenization length
        - 'weighted': Position-weighted pooling
        - 'lemma_replacement': Replace target with lemma in text before embedding (TokLem)
        """
        import time as time_module
        results = []

        # For lemma_replacement strategy, preprocess items to replace targets with lemmas
        if self.pooling_strategy == "lemma_replacement":
            items = self._preprocess_for_lemma_replacement(items)

        # We process in batches to leverage GPU
        total_batches = (len(items) + batch_size - 1) // batch_size
        # Print to terminal (bypassing Streamlit capture)
        print(f"    [batch_extract] {len(items)} items in {total_batches} batches (size={batch_size})", file=sys.__stdout__)
        sys.__stdout__.flush()

        batch_start_time = time_module.time()

        for i in range(0, len(items), batch_size):
            batch_num = i // batch_size + 1
            elapsed = time_module.time() - batch_start_time

            # Memory monitoring
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                mem_info = f" [GPU: {mem_alloc:.1f}GB alloc, {mem_reserved:.1f}GB reserved]"
            else:
                mem_info = ""

            if batch_num == 1:
                print(f"    [batch_extract] Batch {batch_num}/{total_batches}...{mem_info}", file=sys.__stdout__)
            else:
                avg = elapsed / (batch_num - 1)
                rem = avg * (total_batches - batch_num + 1)
                print(f"    [batch_extract] Batch {batch_num}/{total_batches} (avg {avg:.1f}s, ~{rem:.0f}s left){mem_info}", file=sys.__stdout__)
            sys.__stdout__.flush()

            batch_items = items[i : i + batch_size]
            batch_texts = [x['text'] for x in batch_items]
            batch_targets = [x['targets'] for x in batch_items]
            batch_ids = [x['id'] for x in batch_items]
            # For lemma_replacement, track original tokens
            batch_original_tokens = [x.get('original_token') for x in batch_items]

            # Tokenize with offset mapping for character-to-subword alignment
            encodings = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True
            )

            # Move input tensors to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            # offset_mapping stays on CPU (it's used for alignment lookup)
            offset_mappings = encodings['offset_mapping']

            # Forward pass through the model
            with torch.no_grad():
                if self.mixed_precision_dtype and self.device.type == 'cuda':
                    with torch.autocast('cuda', dtype=self.mixed_precision_dtype):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # hidden_states is a tuple of (num_layers + 1) tensors, each (batch, seq_len, hidden_dim)
            # hidden_states[0] is input embeddings, [1] to [12] are transformer layers (BERT-base)
            hidden_states = outputs.hidden_states

            # Collect word vectors on GPU, transfer in batch at end
            batch_word_vecs = []  # List of tensors on GPU
            batch_metadata = []   # Corresponding metadata

            for doc_idx in range(len(batch_texts)):
                sent_id = batch_ids[doc_idx]
                sent_targets = batch_targets[doc_idx]
                original_token = batch_original_tokens[doc_idx]
                text = batch_texts[doc_idx]
                offset_mapping = offset_mappings[doc_idx].tolist()

                # Extract hidden states for this document only
                # We need to index into hidden_states for this specific document
                doc_hidden_states = tuple(hs[doc_idx:doc_idx+1, :, :] for hs in hidden_states)

                for lemma, pos, w_start, w_end in sent_targets:
                    # Find subword token indices that overlap with the character span
                    token_indices = self._get_subword_indices_for_span(offset_mapping, w_start, w_end)

                    if not token_indices:
                        # No tokens found for this span - skip
                        continue

                    # Extract and combine layers for these token indices
                    # Returns (num_tokens, output_dim)
                    combined = self._combine_layers(doc_hidden_states, token_indices)

                    # Pool subword tokens according to strategy
                    word_vec = self._pool_subwords(combined, lemma=lemma)

                    batch_word_vecs.append(word_vec)

                    # Determine the actual token text for metadata
                    if self.pooling_strategy == "lemma_replacement" and original_token:
                        # For TokLem, use the original token (before replacement)
                        token_text = original_token
                    else:
                        token_text = text[w_start:w_end]

                    batch_metadata.append({
                        "lemma": lemma,
                        "pos": pos,
                        "sentence_id": sent_id,
                        "start_char": w_start,
                        "end_char": w_end,
                        "token": token_text
                    })

            # Batch transfer: stack all vectors and move to CPU once
            if batch_word_vecs:
                stacked = torch.stack(batch_word_vecs)  # (N, dim) on GPU
                numpy_vecs = stacked.detach().cpu().numpy()  # Single transfer
                del stacked  # Free GPU memory

                for idx, meta in enumerate(batch_metadata):
                    meta["vector"] = numpy_vecs[idx]
                    results.append(meta)

            # Clean up GPU memory to prevent slowdown
            del hidden_states, outputs, input_ids, attention_mask
            if batch_word_vecs:
                del batch_word_vecs
            torch.cuda.empty_cache()

        print(f"    [batch_extract] Done. {len(results)} embeddings extracted.", file=sys.__stdout__)
        sys.__stdout__.flush()
        return results

    def get_embeddings(self, samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """
        Legacy/Convenience method for on-the-fly extraction.
        samples is a list of dicts with 'sentence', 'start_char', 'matched_word', 'lemma', 'sentence_id'.
        """
        items = []
        for s in samples:
            items.append({
                'id': s.get('sentence_id', 0),
                'text': s['sentence'],
                'targets': [(s['lemma'], s.get('pos', 'NOUN'), s['start_char'], s['start_char'] + len(s['matched_word']))]
            })
        
        extracted = self.batch_extract(items)
        
        if not extracted:
            return np.array([]), []
            
        embeddings = np.array([x['vector'] for x in extracted])
        # Re-map back to sentences (assuming 1-to-1 from items to extracted for these simple samples)
        # To be safe, we should really track which ones failed.
        valid_sentences = []
        # Create a map for quick lookup
        text_map = {x['id']: x['text'] for x in items}
        for x in extracted:
            valid_sentences.append(text_map.get(x['sentence_id'], "[Missing Text]"))
            
        return embeddings, valid_sentences

    def get_nearest_neighbors(self, vector: np.ndarray, target_word: str = None, k: int = 10, pos_filter: str = None) -> Dict[str, np.ndarray]:
        """
        Finds the nearest semantic neighbors by projecting through the MLM head.
        """
        if self.mlm_model is None:
            return {}

        # Ensure vector is a torch tensor
        vec_tensor = torch.from_numpy(vector).to(self.device).float()
        
        # Get target lemma for filtering
        target_lemma = None
        if target_word and self.filter_nlp:
            target_lemma = self.filter_nlp(target_word.lower())[0].lemma_

        # Identify the head module dynamically
        head_module = None
        if hasattr(self.mlm_model, 'cls'): # BERT
            head_module = self.mlm_model.cls
        elif hasattr(self.mlm_model, 'head'): # ModernBERT / RoBERTa
            head_module = self.mlm_model.head
        elif hasattr(self.mlm_model, 'lm_head'): # DistilBERT / Others
            head_module = self.mlm_model.lm_head
        
        if head_module is None:
            # Fallback: try using get_output_embeddings directly
            decoder = self.mlm_model.get_output_embeddings()
            if decoder:
                logits = torch.matmul(vec_tensor, decoder.weight.T)
                if decoder.bias is not None:
                    logits += decoder.bias
                logits = logits.unsqueeze(0).unsqueeze(0)
            else:
                return {}
        else:
            with torch.no_grad():
                # Use mixed precision if available
                if self.mixed_precision_dtype and self.device.type == 'cuda':
                    with torch.autocast('cuda', dtype=self.mixed_precision_dtype):
                        logits = head_module(vec_tensor.unsqueeze(0).unsqueeze(0))
                else:
                    logits = head_module(vec_tensor.unsqueeze(0).unsqueeze(0))

        with torch.no_grad():
            probs = torch.softmax(logits[0, 0], dim=-1)
            top_k_large = torch.topk(probs, k * 30)
            top_ids = top_k_large.indices
            output_embeddings = self.mlm_model.get_output_embeddings()

            # Detect tokenizer style once (outside the loop)
            # RoBERTa/GPT2/ModernBERT use Ġ prefix for word-start tokens
            # SentencePiece (T5, etc.) uses ▁ prefix for word-start tokens
            # BERT uses ## prefix for continuation tokens
            vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else self.tokenizer.vocab
            vocab_tokens = list(vocab.keys())

            # Check across more tokens for reliable detection
            is_roberta_style = any(t.startswith("Ġ") for t in vocab_tokens[:5000])
            is_sentencepiece_style = any(t.startswith("▁") for t in vocab_tokens[:5000])
            is_bert_style = any(t.startswith("##") for t in vocab_tokens[:5000])

            print(f"[DEBUG] Tokenizer style detection: BERT={is_bert_style}, RoBERTa={is_roberta_style}, SentencePiece={is_sentencepiece_style}")

            results = {}
            for token_id in top_ids:
                token_id_scalar = token_id.item()
                token_str = self.tokenizer.decode([token_id_scalar]).strip()

                # Get raw token to check for subword markers
                raw_token = self.tokenizer.convert_ids_to_tokens([token_id_scalar])[0]

                if not token_str or len(token_str) < 2:
                    continue

                # Skip special tokens
                if token_str.startswith("[") or token_str.startswith("<"):
                    continue
                if raw_token in self.tokenizer.all_special_tokens:
                    continue

                # Skip BERT-style continuation tokens (##prefix) - always skip these
                if raw_token.startswith("##"):
                    continue

                # Determine if this is a valid word-start token
                is_word_start = False

                if is_roberta_style and raw_token.startswith("Ġ"):
                    # Definitely a word-start token in RoBERTa-style
                    is_word_start = True
                elif is_sentencepiece_style and raw_token.startswith("▁"):
                    # Definitely a word-start token in SentencePiece-style
                    is_word_start = True
                elif is_bert_style and not is_roberta_style and not is_sentencepiece_style:
                    # Pure BERT-style: if it doesn't start with ##, it's a word-start
                    is_word_start = True
                elif not is_bert_style and not is_roberta_style and not is_sentencepiece_style:
                    # Unknown tokenizer: use re-encoding heuristic
                    test_encoded = self.tokenizer.encode(" " + token_str, add_special_tokens=False)
                    is_word_start = (len(test_encoded) == 1)
                # For hybrid tokenizers (both BERT and RoBERTa markers present):
                # Only accept tokens with explicit word-start markers (Ġ or ▁)
                # Tokens without markers are ambiguous and likely continuations

                if not is_word_start:
                    continue

                # 3. Filter Stopwords (still useful for general noise)
                token_lower = token_str.lower()
                if self.filter_nlp and token_lower in self.filter_nlp.Defaults.stop_words:
                    continue

                if self.filter_nlp:
                    try:
                        cand_doc = self.filter_nlp(token_lower)
                        if not cand_doc or len(cand_doc) == 0:
                            continue

                        cand_token = cand_doc[0]

                        # Filter out non-content POS tags
                        if cand_token.pos_ in ('X', 'PUNCT', 'SYM', 'SPACE'):
                            continue

                        if target_lemma:
                            cand_lemma = cand_token.lemma_
                            if cand_lemma == target_lemma:
                                continue
                            if token_lower in target_word.lower() or target_word.lower() in token_lower:
                                if len(token_str) > 3:
                                    continue

                        if pos_filter and cand_token.pos_ != pos_filter:
                            continue

                    except Exception as e:
                        continue

                token_vec = output_embeddings.weight[token_id].detach().cpu().numpy()
                results[token_str] = token_vec

                if len(results) >= k:
                    break

        return results