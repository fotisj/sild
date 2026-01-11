# Test script for debugging embedding generation

# Test 1: Check if metadata was saved correctly
print("=== Test 1: Checking database metadata ===")
from src.semantic_change.corpus import get_spacy_model_from_db

spacy_t1 = get_spacy_model_from_db('data/corpus_t1.db')
spacy_t2 = get_spacy_model_from_db('data/corpus_t2.db')
print(f"T1 spaCy model: {spacy_t1}")
print(f"T2 spaCy model: {spacy_t2}")

# Test 2: Run batch generation outside Streamlit
print("\n=== Test 2: Running batch embedding generation ===")
from src.semantic_change.embeddings_generation import run_batch_generation

run_batch_generation(
    db_path_t1='data/corpus_t1.db',
    db_path_t2='data/corpus_t2.db',
    model_name='LSX-UniWue/ModernGBERT_1B',
    min_freq=25,
    reset_collections=True,
    test_mode=True
)

print("\n=== Done ===")
