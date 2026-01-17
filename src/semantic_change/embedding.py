import os
# Suppress HuggingFace xet storage warning (falls back to HTTP which works fine)
os.environ["HF_HUB_DISABLE_XET_WARNING"] = "1"

import torch
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Tuple, Dict, Any, Optional
import warnings
import logging

# Suppress huggingface_hub xet warnings
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)


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

    Returns:
        Recommended batch size for embedding extraction.
        - 512 for GPUs with 40GB+ VRAM (A100, L40, etc.)
        - 256 for GPUs with 24GB+ VRAM (RTX 3090/4090, A5000, etc.)
        - 128 for GPUs with 12GB+ VRAM (RTX 3080, etc.)
        - 64 for GPUs with less VRAM or CPU
    """
    if not torch.cuda.is_available():
        return 64

    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        if vram_gb >= 40:
            return 512
        elif vram_gb >= 24:
            return 256
        elif vram_gb >= 12:
            return 128
        else:
            return 64
    except Exception:
        return 128

class Embedder:
    """Base class for embedding generation."""
    def get_embeddings(self, samples: List[Dict[str, str]]) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError

class BertEmbedder(Embedder):
    """
    Generates contextual embeddings using a Transformer model and provides MLM-based decoding.
    Allows configuration of which layers to use and how to combine them.
    Utilizes spacy-transformers for robust linguistic-to-transformer alignment.

    The MLM model and filter model are loaded lazily on first call to get_nearest_neighbors()
    to reduce memory usage and startup time for batch embedding extraction.
    """
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = None,
                 layers: List[int] = None, layer_op: str = 'mean', lang: str = 'en',
                 filter_model: str = None):
        """
        Args:
            model_name: HuggingFace model name.
            device: 'cpu' or 'cuda'.
            layers: List of layer indices to use (e.g., [-1, -2, -3, -4]). Default is [-1].
            layer_op: How to combine layers: 'mean', 'sum', or 'concat'. Default is 'mean'.
            lang: Language code for the blank spaCy model.
            filter_model: spaCy model for neighbor filtering (lemma/pos). If None, uses a default based on lang.
        """
        self.model_name = model_name
        self.lang = lang
        self.filter_model_name = filter_model
        self.device_name = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device_name)

        # Detect optimal mixed precision dtype for this GPU
        self.mixed_precision_dtype = detect_optimal_dtype()

        import sys
        print(f"Loading model '{model_name}' on {self.device_name}...")
        if self.mixed_precision_dtype:
            dtype_name = 'bfloat16' if self.mixed_precision_dtype == torch.bfloat16 else 'float16'
            print(f"  Mixed precision enabled: {dtype_name}")
        sys.stdout.flush()

        # 1. Initialize spaCy pipeline with transformer
        try:
            # Set device for spacy-transformers before loading
            if self.device.type == "cuda":
                print(f"  Requesting GPU {self.device.index if self.device.index is not None else 0}...")
                sys.stdout.flush()
                spacy.require_gpu(self.device.index if self.device.index is not None else 0)

            print(f"  Creating spaCy pipeline...")
            sys.stdout.flush()
            self.nlp = spacy.blank(lang)
            # Use spacy-transformers to add the transformer component
            config = {
                "model": {
                    "@architectures": "spacy-transformers.TransformerModel.v3",
                    "name": model_name,
                    "tokenizer_config": {"use_fast": True},
                    "transformer_config": {"output_hidden_states": True}
                }
            }
            print(f"  Adding transformer component (downloading/loading model weights)...")
            sys.stdout.flush()
            self.nlp.add_pipe("transformer", config=config)

            # Initialize the pipeline (crucial for loading the actual model/tokenizer)
            print(f"  Initializing pipeline...")
            sys.stdout.flush()
            self.nlp.initialize()
            print(f"  spaCy pipeline ready.")
            sys.stdout.flush()

        except Exception as e:
            print(f"Error initializing spacy-transformers: {e}")
            sys.stdout.flush()
            raise

        # 2. Tokenizer is needed for decoding, load it now
        print(f"  Loading tokenizer...")
        sys.stdout.flush()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  Tokenizer loaded.")
        sys.stdout.flush()

        # Lazy-loaded components (initialized on first use in get_nearest_neighbors)
        self._mlm_model = None  # Will be loaded on first neighbor query
        self._filter_nlp = None  # Will be loaded on first neighbor query
        self._mlm_load_failed = False  # Track if MLM loading already failed

        self.layers = layers if layers is not None else [-1]
        self.layer_op = layer_op.lower()

        print(f"BertEmbedder initialization complete for '{model_name}'.")
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

    def _combine_layers(self, trf_data) -> torch.Tensor:
        """
        Extracts and combines hidden states from trf_data.
        Optimized to stay on GPU when using cupy arrays (avoids GPU→CPU→GPU round-trip).
        """
        # trf_data.tensors[0] is (num_spans, seq_len, dim)
        arr = trf_data.tensors[0]
        num_spans, seq_len, dim = arr.shape

        # Multi-layer warning (we only support last layer currently)
        if self.layers != [-1] and len(self.layers) > 0:
            warnings.warn("Multi-layer combination via spacy-transformers is currently limited to the last layer. Using last layer.")

        # Handle cupy arrays (GPU) - use DLPack for zero-copy transfer to torch
        if hasattr(arr, "__dlpack__"):
            # Reshape on cupy array first (stays on GPU)
            flat_arr = arr.reshape(num_spans * seq_len, dim)
            # Zero-copy transfer via DLPack (requires torch >= 1.10)
            return torch.from_dlpack(flat_arr)
        elif hasattr(arr, "get"):
            # Fallback for older cupy without DLPack support
            arr = arr.get()
            flat_arr = arr.reshape(num_spans * seq_len, dim)
            return torch.from_numpy(flat_arr).to(self.device)
        else:
            # numpy array (CPU)
            flat_arr = arr.reshape(num_spans * seq_len, dim)
            return torch.from_numpy(flat_arr).to(self.device)

    def batch_extract(self, items: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Extracts embeddings for specific words using spacy-transformers alignment.
        Optimized to batch GPU→CPU transfers.
        Uses mixed precision (bf16/fp16) when available for ~2x speedup.
        """
        results = []

        # We process in batches to leverage GPU
        for i in range(0, len(items), batch_size):
            batch_items = items[i : i + batch_size]
            batch_texts = [x['text'] for x in batch_items]
            batch_targets = [x['targets'] for x in batch_items]
            batch_ids = [x['id'] for x in batch_items]

            # 1. Run spaCy pipeline (includes transformer)
            # pipe() handles batching internally
            # Use mixed precision if available (bf16 for datacenter GPUs, fp16 for consumer)
            if self.mixed_precision_dtype and self.device.type == 'cuda':
                with torch.autocast('cuda', dtype=self.mixed_precision_dtype):
                    docs = list(self.nlp.pipe(batch_texts, batch_size=len(batch_texts)))
            else:
                docs = list(self.nlp.pipe(batch_texts, batch_size=len(batch_texts)))

            # Collect word vectors on GPU, transfer in batch at end
            batch_word_vecs = []  # List of tensors on GPU
            batch_metadata = []   # Corresponding metadata

            for doc_idx, doc in enumerate(docs):
                sent_id = batch_ids[doc_idx]
                sent_targets = batch_targets[doc_idx]

                trf_data = doc._.trf_data
                # Combined vectors for all transformer tokens in this doc
                combined_vectors = self._combine_layers(trf_data)  # (seq_len, dim)

                # Alignment map: doc._.trf_data.align
                # align[i] gives indices of transformer tokens for the i-th spacy token
                align = trf_data.align

                for lemma, pos, w_start, w_end in sent_targets:
                    # Find the spacy token(s) that match the char span [w_start, w_end)
                    span = doc.char_span(w_start, w_end, alignment_mode="expand")
                    if span is None:
                        continue

                    # Collect all transformer token indices for all spacy tokens in the span
                    trf_indices = []
                    for token in span:
                        trf_indices.extend(align[token.i].data.flatten())

                    if trf_indices:
                        # Deduplicate indices using set - faster than sorted(list(set()))
                        trf_indices = list(set(trf_indices))

                        # Extract vectors and pool on GPU
                        sub_vectors = combined_vectors[trf_indices]  # (num_subtokens, dim)
                        word_vec = torch.mean(sub_vectors, dim=0)  # Stay on GPU

                        batch_word_vecs.append(word_vec)
                        batch_metadata.append({
                            "lemma": lemma,
                            "pos": pos,
                            "sentence_id": sent_id,
                            "start_char": w_start,
                            "end_char": w_end
                        })

            # Batch transfer: stack all vectors and move to CPU once
            if batch_word_vecs:
                stacked = torch.stack(batch_word_vecs)  # (N, dim) on GPU
                numpy_vecs = stacked.detach().cpu().numpy()  # Single transfer

                for idx, meta in enumerate(batch_metadata):
                    meta["vector"] = numpy_vecs[idx]
                    results.append(meta)

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