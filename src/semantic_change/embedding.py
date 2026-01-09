import torch
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Tuple, Dict, Any, Optional
import warnings

class Embedder:
    """Base class for embedding generation."""
    def get_embeddings(self, samples: List[Dict[str, str]]) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError

class BertEmbedder(Embedder):
    """
    Generates contextual embeddings using a Transformer model and provides MLM-based decoding.
    Allows configuration of which layers to use and how to combine them.
    Utilizes spacy-transformers for robust linguistic-to-transformer alignment.
    """
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = None, 
                 layers: List[int] = None, layer_op: str = 'mean', lang: str = 'en'):
        """
        Args:
            model_name: HuggingFace model name.
            device: 'cpu' or 'cuda'.
            layers: List of layer indices to use (e.g., [-1, -2, -3, -4]). Default is [-1].
            layer_op: How to combine layers: 'mean', 'sum', or 'concat'. Default is 'mean'.
            lang: Language code for the blank spaCy model.
        """
        self.device_name = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device_name)
        
        print(f"Loading model '{model_name}' on {self.device_name}...")
        
        # 1. Initialize spaCy pipeline with transformer
        try:
            # Set device for spacy-transformers before loading
            if self.device.type == "cuda":
                spacy.require_gpu(self.device.index if self.device.index is not None else 0)
            
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
            self.nlp.add_pipe("transformer", config=config)
            
            # Initialize the pipeline (crucial for loading the actual model/tokenizer)
            self.nlp.initialize()
            
        except Exception as e:
            print(f"Error initializing spacy-transformers: {e}")
            raise

        # 2. For nearest neighbors, we still need the AutoModelForMaskedLM to access the head
        # and the tokenizer for decoding.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.mlm_model.eval()
        
        self.layers = layers if layers is not None else [-1]
        self.layer_op = layer_op.lower()

        # Load a standard spaCy model for neighbor filtering (lemma/pos)
        try:
            # We use a non-transformer model for fast linguistic filtering of neighbors
            self.filter_nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
        except OSError:
            print("Warning: spaCy model 'en_core_web_lg' not found. Neighbor filtering might be limited.")
            self.filter_nlp = None

        # Load NLTK English word list for validating real words
        self.english_words = None
        try:
            import nltk
            try:
                self.english_words = set(w.lower() for w in nltk.corpus.words.words())
                print(f"Loaded NLTK word list with {len(self.english_words)} words")
            except LookupError:
                print("Downloading NLTK words corpus...")
                nltk.download('words')
                self.english_words = set(w.lower() for w in nltk.corpus.words.words())
        except ImportError:
            print("Warning: NLTK not available. Word validation will be limited.")

    def _combine_layers(self, trf_data) -> torch.Tensor:
        """
        Extracts and combines hidden states from trf_data.
        """
        # trf_data.tensors[0] is (num_spans, seq_len, dim)
        arr = trf_data.tensors[0]
        
        # Handle cupy arrays if on GPU
        if hasattr(arr, "get"):
            arr = arr.get()
            
        # Reshape spans to a flat sequence: (num_spans * seq_len, dim)
        # This matches the indices in the alignment map.
        num_spans, seq_len, dim = arr.shape
        flat_arr = arr.reshape(num_spans * seq_len, dim)
        
        # If we only want the last layer (default)
        if self.layers == [-1] or len(self.layers) == 0:
            return torch.from_numpy(flat_arr).to(self.device)

        # If multiple layers are requested, we currently only support the last layer
        # due to the complexity of slicing model_output.hidden_states across spans.
        if self.layers != [-1]:
            warnings.warn("Multi-layer combination via spacy-transformers is currently limited to the last layer. Using last layer.")
            
        return torch.from_numpy(flat_arr).to(self.device)

    def batch_extract(self, items: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Extracts embeddings for specific words using spacy-transformers alignment.
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
            docs = list(self.nlp.pipe(batch_texts, batch_size=len(batch_texts)))
            
            for doc_idx, doc in enumerate(docs):
                sent_id = batch_ids[doc_idx]
                sent_targets = batch_targets[doc_idx]
                
                trf_data = doc._.trf_data
                # Combined vectors for all transformer tokens in this doc
                combined_vectors = self._combine_layers(trf_data) # (seq_len, dim)
                
                # Alignment map: doc._.trf_data.align
                # align[i] gives indices of transformer tokens for the i-th spacy token
                align = trf_data.align
                
                for lemma, w_start, w_end in sent_targets:
                    # Find the spacy token(s) that match the char span [w_start, w_end)
                    # Use doc.char_span to find the token indices
                    span = doc.char_span(w_start, w_end, alignment_mode="expand")
                    if span is None:
                        continue
                    
                    # Collect all transformer token indices for all spacy tokens in the span
                    trf_indices = []
                    for token in span:
                        trf_indices.extend(align[token.i].data.flatten())
                    
                    if trf_indices:
                        # Deduplicate indices
                        trf_indices = sorted(list(set(trf_indices)))
                        
                        # Extract vectors
                        sub_vectors = combined_vectors[trf_indices] # (num_subtokens, dim)
                        
                        # Pool (mean) to get word embedding
                        word_vec = torch.mean(sub_vectors, dim=0).detach().cpu().numpy()
                        
                        results.append({
                            "lemma": lemma,
                            "vector": word_vec,
                            "sentence_id": sent_id,
                            "start_char": w_start,
                            "end_char": w_end
                        })
        
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
                'targets': [(s['lemma'], s['start_char'], s['start_char'] + len(s['matched_word']))]
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
                # Some heads expect (batch, seq, dim)
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

                # 2. Validate against English dictionary (most reliable filter)
                token_lower = token_str.lower()
                if self.english_words:
                    if token_lower not in self.english_words:
                        continue

                # 3. Filter Stopwords
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
