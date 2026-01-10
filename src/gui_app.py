import streamlit as st
import os
import sys
import contextlib
import json
import glob
import time
import numpy as np

# Import analysis modules path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

CONFIG_FILE = "config.json"
OUTPUT_DIR = "output"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Helper Functions ---

@st.cache_resource
def get_embedder(model_name, layers=None, layer_op='mean', lang='en'):
    """
    Loads and caches the BERT Embedder model.
    Lazy import to speed up app startup.
    """
    from semantic_change.embedding import BertEmbedder
    return BertEmbedder(model_name=model_name, layers=layers, layer_op=layer_op, lang=lang)

def get_available_models():
    """
    Scans ChromaDB to find which models have been processed.
    Returns a list of unique model names.
    """
    try:
        from semantic_change.vector_store import VectorStore
        v_store = VectorStore(persistence_path="data/chroma_db")
        collections = v_store.list_collections()
        models = set()
        for c in collections:
            # Expected format: embeddings_t1_{model_safe}
            if c.startswith("embeddings_t1_"):
                # Extract safe model name
                safe_name = c.replace("embeddings_t1_", "")
                # We can't easily reverse safe_name back to original (e.g. _ -> - or /)
                # But we can display the safe name, or if we stored mapping, use that.
                # For now, let's try to beautify or just list it.
                # Actually, the user input "model_name" is needed for loading the model for inference.
                # So we need the EXACT HuggingFace ID. 
                # Reversing the safe name is lossy.
                # Workaround: Check against the currently configured model?
                # Better: Just list the safe names and let the user select?
                # NO, we need to load the model.
                # Compromise: We will just list the collections as "ID: ..."
                # and relying on the user to pick the one that matches.
                # OR better: Store a mapping in a local json file?
                models.add(safe_name)
        return sorted(list(models))
    except Exception:
        return []

def load_config():
    default_config = {
        "data_dir": "data",
        "input_dir_t1": "data_gutenberg/1800",
        "input_dir_t2": "data_gutenberg/1900",
        "period_t1_label": "1800",
        "period_t2_label": "1900",
        "spacy_model": "en_core_web_sm",
        "model_name": "bert-base-uncased",
        "target_word": "current",
        "k_neighbors": 10,
        "min_cluster_size": 3,
        "n_clusters": 3,
        "wsi_algorithm": "hdbscan",
        "pos_filter": "None",
        # Dimensionality reduction settings
        "clustering_reduction": "None",  # None, pca, umap, tsne
        "clustering_n_components": 50,
        "viz_reduction": "pca",  # pca, umap, tsne
        # Legacy (kept for backwards compatibility)
        "use_umap": True,
        "umap_n_components": 50,
        "n_samples": 50,
        "viz_max_instances": 100,
        "min_freq": 25,
        "layers": [-1],
        "layer_op": "mean",
        "lang": "en",
        "context_window": 0  # 0 means sentence only
    }
    
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            loaded_config = json.load(f)
            for key, val in default_config.items():
                if key not in loaded_config:
                    loaded_config[key] = val
            return loaded_config
    return default_config

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

@contextlib.contextmanager
def capture_stdout(output_func):
    """Captures stdout and sends it to a Streamlit output function."""
    class StreamlitWriter:
        def write(self, text):
            output_func(text)
        def flush(self):
            pass
    old_stdout = sys.stdout
    sys.stdout = StreamlitWriter()
    try:
        yield
    finally:
        sys.stdout = old_stdout

# --- Main App Configuration ---

st.set_page_config(page_title="Semantic Change Analysis", layout="wide", page_icon="📚")

if "config" not in st.session_state:
    st.session_state.config = load_config()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["Analysis Dashboard", "Data Ingestion", "Embeddings & Models", "Corpus Reports"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Global Settings**")
st.session_state.config["data_dir"] = st.sidebar.text_input("Data Output Directory", value=st.session_state.config["data_dir"])

if st.sidebar.button("Save Configuration"):
    save_config(st.session_state.config)
    st.sidebar.success("Config saved!")

# Period labels (configurable in sidebar)
st.sidebar.markdown("**Time Period Labels**")
period_t1_label = st.sidebar.text_input("Period 1 Label", value=st.session_state.config.get("period_t1_label", "1800"))
period_t2_label = st.sidebar.text_input("Period 2 Label", value=st.session_state.config.get("period_t2_label", "1900"))
st.session_state.config["period_t1_label"] = period_t1_label
st.session_state.config["period_t2_label"] = period_t2_label

# Check DB status (fixed filenames)
db_t1 = os.path.join(st.session_state.config["data_dir"], "corpus_t1.db")
db_t2 = os.path.join(st.session_state.config["data_dir"], "corpus_t2.db")
dbs_exist = os.path.exists(db_t1) and os.path.exists(db_t2)

if dbs_exist:
    st.sidebar.success("✅ Databases Ready")
else:
    st.sidebar.warning("⚠️ Databases Missing")


# --- Page: Data Ingestion ---
if page == "Data Ingestion":
    st.title("🗄️ Corpus Ingestion")
    st.markdown("""
    Process raw text files into a structured SQLite database.
    **This step must be done once before analysis.**
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Corpus: {period_t1_label}")
        input_t1 = st.text_input(f"Input Directory ({period_t1_label})", value=st.session_state.config.get("input_dir_t1", "data_gutenberg/1800"))
        st.session_state.config["input_dir_t1"] = input_t1

    with col2:
        st.subheader(f"Corpus: {period_t2_label}")
        input_t2 = st.text_input(f"Input Directory ({period_t2_label})", value=st.session_state.config.get("input_dir_t2", "data_gutenberg/1900"))
        st.session_state.config["input_dir_t2"] = input_t2

    st.markdown("### Settings")
    spacy_model = st.selectbox(
        "SpaCy Model (Lemmatization)",
        ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
        index=0
    )
    st.session_state.config["spacy_model"] = spacy_model

    st.info(f"Output files will be saved to: `{db_t1}` and `{db_t2}`")

    if st.button("Run Ingestion Process", type="primary"):
        save_config(st.session_state.config)

        if not os.path.exists(input_t1) or not os.path.exists(input_t2):
            st.error("One or both input directories do not exist.")
        else:
            log_container = st.empty()
            logs = []
            def update_ingest_log(text):
                logs.append(text)
                log_container.code("".join(logs[-30:]))  # Keep last 30 lines

            start_time = time.time()
            ingest_pbar = st.progress(0)
            ingest_status = st.empty()
            
            def ingest_progress_callback(current, total, desc):
                if total > 0:
                    progress = min(current / total, 1.0)
                    ingest_pbar.progress(progress)
                    ingest_status.text(f"{desc} ({current}/{total})")

            with st.spinner("Ingesting corpora... This may take a while."):
                try:
                    from semantic_change.ingestor import Ingestor
                    from semantic_change.corpus import Corpus
                    from semantic_change.reporting import generate_comparison_report

                    with capture_stdout(update_ingest_log):
                        ingestor = Ingestor(model=spacy_model)

                        print(f"--- Processing {period_t1_label} from {input_t1} ---")
                        ingestor.preprocess_corpus(input_t1, db_t1, progress_callback=ingest_progress_callback)

                        print(f"--- Processing {period_t2_label} from {input_t2} ---")
                        ingestor.preprocess_corpus(input_t2, db_t2, progress_callback=ingest_progress_callback)

                        # Generate initial report
                        print("--- Generating Initial Comparison Report ---")
                        c_t1 = Corpus(period_t1_label, input_t1, db_t1)
                        c_t2 = Corpus(period_t2_label, input_t2, db_t2)
                        generate_comparison_report(c_t1, c_t2, top_n=50)
                        
                    elapsed = time.time() - start_time
                    st.success(f"Ingestion completed in {elapsed/60:.2f} minutes!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    st.exception(e)

    st.markdown("---")
    st.subheader("🗑️ Danger Zone")
    st.warning("These actions will delete processed data.")
    
    if st.button("Wipe SQLite Databases", help="Deletes the corpus_t1.db and corpus_t2.db files."):
        try:
            if os.path.exists(db_t1): os.remove(db_t1)
            if os.path.exists(db_t2): os.remove(db_t2)
            st.success("SQLite databases deleted.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to delete databases: {e}")


# --- Page: Embeddings & Models ---
elif page == "Embeddings & Models":
    st.title("🧬 Embeddings & Models")
    
    st.markdown("""
    Manage the LLM models used for semantic analysis. You can generate embeddings for multiple models and switch between them in the Dashboard.
    """)
    
    tab_new, tab_manage = st.tabs(["Create New Embeddings", "Manage Existing"])
    
    # --- Tab: Create New ---
    with tab_new:
        st.subheader("Generate Embeddings")
        st.info("This process uses the chosen model to compute vector representations for the words in your ingested corpus.")
        
        col1, col2 = st.columns(2)
        with col1:
            # Common models list to prevent typos
            common_models = [
                "bert-base-uncased",
                "roberta-base",
                "google/modernbert-base-uncased",
                "distilbert-base-uncased",
                "xlm-roberta-base",
                "Other (Custom)"
            ]
            
            # Try to match current config to list
            current_model_val = st.session_state.config["model_name"]
            default_index = 0
            if current_model_val in common_models:
                default_index = common_models.index(current_model_val)
            else:
                default_index = common_models.index("Other (Custom)")
            
            selected_model_option = st.selectbox("Select Model", common_models, index=default_index)
            
            if selected_model_option == "Other (Custom)":
                new_model_name = st.text_input("Hugging Face Model ID", value=current_model_val, help="e.g. bert-large-uncased")
            else:
                new_model_name = selected_model_option
                
            st.session_state.config["model_name"] = new_model_name
            
            min_freq_val = st.number_input("Minimum Frequency", min_value=5, value=st.session_state.config.get("min_freq", 25), step=5)
            st.session_state.config["min_freq"] = min_freq_val
            
        with col2:
            custom_words_input = st.text_area("Custom Words (optional)", 
                                              placeholder="e.g. apple, banana\n(comma or newline separated)",
                                              help="Add specific words to the batch processing list, regardless of frequency.")

        st.markdown("#### Advanced Model Config")
        with st.expander("Layer Selection & Combination"):
            layer_options = {
                "Last Layer (-1)": -1,
                "2nd Last Layer (-2)": -2,
                "3rd Last Layer (-3)": -3,
                "4th Last Layer (-4)": -4
            }
            default_layers = []
            current_layers = st.session_state.config.get("layers", [-1])
            for label, val in layer_options.items():
                if val in current_layers:
                    default_layers.append(label)
                    
            selected_labels = st.multiselect("Layers to Use", list(layer_options.keys()), default=default_layers)
            st.session_state.config["layers"] = [layer_options[l] for l in selected_labels]
            
            combo_options = ["Mean", "Sum", "Concat"]
            current_op = st.session_state.config.get("layer_op", "mean").title()
            if current_op not in combo_options: current_op = "Mean"
            selected_op = st.selectbox("Layer Combination Method", combo_options, index=combo_options.index(current_op))
            st.session_state.config["layer_op"] = selected_op.lower()

        if st.button("Start Batch Process", type="primary"):
            save_config(st.session_state.config)
            
            # Parse custom words
            custom_words = []
            if custom_words_input:
                raw_words = custom_words_input.replace('\n', ',').split(',')
                custom_words = [w.strip() for w in raw_words if w.strip()]
            
            batch_log = st.empty()
            b_logs = []
            def update_batch_logs(text):
                if not text.startswith('\r') and "%" not in text:
                    b_logs.append(text)
                    batch_log.code("".join(b_logs[-20:]))

            pbar = st.progress(0)
            pbar_text = st.empty()
            
            def progress_callback(current, total, desc):
                if total > 0:
                    progress = min(current / total, 1.0)
                    pbar.progress(progress)
                    pbar_text.text(f"{desc}: {current}/{total} ({int(progress*100)}%)")

            with st.spinner(f"Generating embeddings for {new_model_name}..."):
                try:
                    from run_batch_analysis import run_batch_process
                    
                    # Note: We always reset specific collections for *this* model
                    # to ensure fresh start for this specific run.
                    # Or we could append? For now, let's use the reset flag to be safe for this model.
                    # But wait, reset deletes collection by name.
                    # Yes, run_batch_process calculates name from model.
                    
                    with capture_stdout(update_batch_logs):
                        run_batch_process(
                            db_path_t1=db_t1,
                            db_path_t2=db_t2,
                            model_name=new_model_name,
                            min_freq=min_freq_val,
                            additional_words=custom_words,
                            progress_callback=progress_callback,
                            reset_collections=True 
                        )
                    st.success("Batch Processing Complete!")
                    pbar.progress(1.0)
                except Exception as e:
                    st.error(f"Batch process failed: {e}")

    # --- Tab: Manage ---
    with tab_manage:
        st.subheader("Existing Embedding Sets")
        available_models = get_available_models()
        
        if not available_models:
            st.info("No embeddings found in the database.")
        else:
            st.write(f"Found {len(available_models)} processed models:")
            for m in available_models:
                col_m1, col_m2 = st.columns([3, 1])
                col_m1.markdown(f"**{m}**")
                if col_m2.button(f"Delete", key=f"del_{m}"):
                    try:
                        from semantic_change.vector_store import VectorStore
                        v_store = VectorStore(persistence_path="data/chroma_db")
                        v_store.delete_collection(f"embeddings_t1_{m}")
                        v_store.delete_collection(f"embeddings_t2_{m}")
                        st.success(f"Deleted embeddings for {m}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")


# --- Page: Corpus Reports ---
elif page == "Corpus Reports":
    st.title("📊 Corpus Statistics")
    
    if not dbs_exist:
        st.error("Please run Ingestion first.")
    else:
        st.write("Compare word frequencies between the two time periods.")
        report_top_n = st.number_input("Top N Shared Words", value=30, min_value=10)
        
        if st.button("Generate Comparison Report"):
            with st.spinner("Generating Report..."):
                try:
                    from semantic_change.reporting import generate_comparison_report
                    from semantic_change.corpus import CorpusManager

                    manager = CorpusManager()
                    manager.add_corpus(period_t1_label, os.path.dirname(db_t1), db_t1)
                    manager.add_corpus(period_t2_label, os.path.dirname(db_t2), db_t2)

                    report_path = os.path.join(OUTPUT_DIR, "processing_report.md")

                    markdown_content = generate_comparison_report(
                        manager.get_corpus(period_t1_label),
                        manager.get_corpus(period_t2_label),
                        top_n=report_top_n,
                        output_path=report_path
                    )
                    st.markdown(markdown_content)
                    st.success(f"Report saved to {report_path}")
                except Exception as e:
                    st.error(f"Error: {e}")
                    
        report_path = os.path.join(OUTPUT_DIR, "processing_report.md")
        if os.path.exists(report_path):
            st.markdown("---")
            st.subheader("Existing Report")
            with open(report_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())


# --- Page: Analysis Dashboard (Main) ---
elif page == "Analysis Dashboard":
    st.title("🔎 Semantic Analysis")
    
    if not dbs_exist:
        st.error("Databases missing. Please go to 'Data Ingestion' first.")
    else:
        # Check for available models
        available_models = get_available_models()
        
        if not available_models:
            st.warning("⚠️ No embeddings found. Please go to the **Embeddings & Models** tab to generate them first.")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Parameters")
                
                # Model Selection (Selector instead of Text Input)
                # Try to find current config model in list
                current_model_safe = st.session_state.config["model_name"].replace("/", "_").replace("-", "_")
                
                # We have a mismatch: available_models are "safe names" (bert_base_uncased), 
                # but run_single_analysis needs real names (bert-base-uncased).
                # Since we can't perfectly reverse, we rely on the user to pick the safe name, 
                # AND we need to pass that safe name to main.py.
                # BUT main.py expects a real model name to load the tokenizer/model for fallback.
                # This is tricky.
                # SOLUTION: For this prototype, we assume the user knows the mapping or we just pass the 
                # safe name as the "model name". BertEmbedder might fail if we pass "bert_base_uncased" 
                # to from_pretrained.
                # Ideally, we should store metadata about the model name in ChromaDB collection metadata 
                # or a separate config file.
                # Fallback: In this UI, we just ask the user to type the real model name again if it's not in config?
                # OR: We rely on the "model_name" stored in config.json.
                
                # Let's try to map safe names back if they are common, or just show the safe name 
                # and assume the user hasn't changed the "Model Name" input field in the other tab.
                
                # Better UX: Show the "safe names" in the dropdown. 
                # When selected, we might not know the exact HuggingFace ID to instantiate the tokenizer.
                # However, for Visualization (Chroma Neighbors), we DO NOT need the model if we don't do MLM fallback!
                # But run_single_analysis instantiates BertEmbedder for MLM fallback.
                # Let's add a text input "HuggingFace ID for Fallback" pre-filled with config?
                # Or just use the config.
                
                selected_safe_model = st.selectbox(
                    "Select Embedding Set", 
                    available_models,
                    index=0 if available_models else None,
                    help="Select the pre-computed embeddings to use."
                )
                
                # If the user selects a set, we ideally want to update the config's model name
                # so that the embedder can be loaded. But we don't know the real name.
                # Let's hope the user hasn't changed the "Embeddings" tab model input.
                # For now, we will use the text input from the config as the "Real Model Name" 
                # and just use the selected collection for data.
                # This relies on the user keeping them in sync.
                # To make it safer:
                
                st.info(f"Using embeddings from: {selected_safe_model}")
                real_model_name = st.text_input("Confirm Model ID (for Tokenizer)", value=st.session_state.config["model_name"])
                
                # Check consistency
                if real_model_name.replace("/", "_").replace("-", "_") != selected_safe_model:
                    st.warning("⚠️ Model ID does not match the selected embedding set! Neighbor fallback might fail.")

                target_word = st.text_input("Target Word", value=st.session_state.config["target_word"])
                
                # POS Filter
                pos_options = ["None", "NOUN", "VERB", "ADJ"]
                current_pos = st.session_state.config.get("pos_filter", "None")
                if current_pos not in pos_options: current_pos = "None"
                pos_filter = st.selectbox("POS Filter", options=pos_options, index=pos_options.index(current_pos))
                
                n_samples = st.number_input("Samples per Period", min_value=10, max_value=2000, value=st.session_state.config["n_samples"])
                
                wsi_algo = st.selectbox(
                    "Clustering Algorithm", 
                    options=["hdbscan", "kmeans", "spectral", "agglomerative"],
                    index=["hdbscan", "kmeans", "spectral", "agglomerative"].index(st.session_state.config["wsi_algorithm"])
                )
                
                # Dynamic Clustering Params
                min_cluster = 3
                n_clusters = 3
                if wsi_algo == "hdbscan":
                    min_cluster = st.number_input("Min Cluster Size", min_value=2, value=st.session_state.config["min_cluster_size"])
                    st.session_state.config["min_cluster_size"] = min_cluster
                else:
                    n_clusters = st.number_input("Number of Clusters (k)", min_value=2, value=st.session_state.config["n_clusters"])
                    st.session_state.config["n_clusters"] = n_clusters

                st.markdown("##### Dimensionality Reduction")
                # Pre-clustering reduction
                clust_reduction_options = ["None", "pca", "umap", "tsne"]
                current_clust_red = st.session_state.config.get("clustering_reduction", "None")
                if current_clust_red not in clust_reduction_options:
                    current_clust_red = "None"
                clustering_reduction = st.selectbox(
                    "Pre-clustering Reduction",
                    options=clust_reduction_options,
                    index=clust_reduction_options.index(current_clust_red),
                    help="Reduce dimensions before clustering. Helps with high-dim embeddings (~768d)."
                )
                if clustering_reduction != "None":
                    clustering_n_components = st.number_input(
                        "Clustering Reduction Dims",
                        min_value=2, max_value=200,
                        value=st.session_state.config.get("clustering_n_components", 50)
                    )
                else:
                    clustering_n_components = st.session_state.config.get("clustering_n_components", 50)

                # Visualization reduction
                viz_reduction_options = ["pca", "umap", "tsne"]
                current_viz_red = st.session_state.config.get("viz_reduction", "pca")
                if current_viz_red not in viz_reduction_options:
                    current_viz_red = "pca"
                viz_reduction = st.selectbox(
                    "Visualization Reduction",
                    options=viz_reduction_options,
                    index=viz_reduction_options.index(current_viz_red),
                    help="Method to reduce to 2D for plotting."
                )

                k_neighbors = st.number_input("Neighbors (k)", min_value=1, value=st.session_state.config["k_neighbors"])

                context_window = st.slider("Context Window (chars around word)", min_value=0, max_value=5000, value=st.session_state.config.get("context_window", 0), help="0 means use the sentence from DB. >0 reads raw file around the word.")

                st.markdown("##### Contextual Neighbors (MLM Aggregation)")
                n_top_sentences = st.number_input("Sentences to sample", min_value=1, max_value=50, value=st.session_state.config.get("n_top_sentences", 10))
                k_per_sentence = st.number_input("Predictions per sentence", min_value=1, max_value=20, value=st.session_state.config.get("k_per_sentence", 6))

                # Save state
                st.session_state.config["target_word"] = target_word
                st.session_state.config["pos_filter"] = pos_filter
                st.session_state.config["clustering_reduction"] = clustering_reduction
                st.session_state.config["clustering_n_components"] = clustering_n_components
                st.session_state.config["viz_reduction"] = viz_reduction
                st.session_state.config["n_samples"] = n_samples
                st.session_state.config["wsi_algorithm"] = wsi_algo
                st.session_state.config["k_neighbors"] = k_neighbors
                st.session_state.config["context_window"] = context_window
                st.session_state.config["n_top_sentences"] = n_top_sentences
                st.session_state.config["k_per_sentence"] = k_per_sentence
                # Update model name to the one typed in confirmation
                st.session_state.config["model_name"] = real_model_name

                run_btn = st.button("Run Analysis", type="primary")

            with col2:
                log_area = st.empty()

            if run_btn:
                save_config(st.session_state.config)
                final_pos = None if pos_filter == "None" else pos_filter
                
                logs = []
                def update_logs(text):
                    logs.append(text)
                    log_area.code("".join(logs))

                with st.spinner(f"Analyzing..."):
                    try:
                        from main import run_single_analysis
                        embedder = get_embedder(
                            st.session_state.config["model_name"],
                            layers=st.session_state.config.get("layers", [-1]),
                            layer_op=st.session_state.config.get("layer_op", "mean"),
                            lang=st.session_state.config.get("lang", "en")
                        )
                        
                        clust_red = clustering_reduction if clustering_reduction != "None" else None

                        with capture_stdout(update_logs):
                            run_single_analysis(
                                target_word=target_word,
                                db_path_t1=db_t1,
                                db_path_t2=db_t2,
                                period_t1_label=period_t1_label,
                                period_t2_label=period_t2_label,
                                model_name=st.session_state.config["model_name"],
                                k_neighbors=k_neighbors,
                                min_cluster_size=min_cluster,
                                n_clusters=n_clusters,
                                wsi_algorithm=wsi_algo,
                                pos_filter=final_pos,
                                clustering_reduction=clust_red,
                                clustering_n_components=clustering_n_components,
                                viz_reduction=viz_reduction,
                                n_samples=n_samples,
                                viz_max_instances=st.session_state.config["viz_max_instances"],
                                embedder=embedder,
                                context_window=context_window,
                                n_top_sentences=n_top_sentences,
                                k_per_sentence=k_per_sentence
                            )
                        
                        st.success("Analysis Complete!")
                        
                        st.subheader("Visualizations")
                        tp_path = os.path.join(OUTPUT_DIR, "time_period.html")
                        if os.path.exists(tp_path):
                            st.markdown("### ⏳ Time Period Clustering")
                            with open(tp_path, 'r', encoding='utf-8') as f:
                                st.components.v1.html(f.read(), height=600, scrolling=True)
                        
                        sc_path = os.path.join(OUTPUT_DIR, "sense_clusters.html")
                        if os.path.exists(sc_path):
                            st.markdown("### 🧩 Sense Clusters")
                            with open(sc_path, 'r', encoding='utf-8') as f:
                                st.components.v1.html(f.read(), height=600, scrolling=True)

                        stc_path = os.path.join(OUTPUT_DIR, "sense_time_combined.html")
                        if os.path.exists(stc_path):
                            st.markdown("### 🎨 Sense × Time (Combined)")
                            with open(stc_path, 'r', encoding='utf-8') as f:
                                st.components.v1.html(f.read(), height=600, scrolling=True)

                        neighbor_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "neighbors_cluster_*.html")))
                        if neighbor_files:
                            st.markdown("### 🕸️ Semantic Neighbors")
                            for nf in neighbor_files:
                                cluster_name = os.path.basename(nf).replace("neighbors_", "").replace(".html", "").replace("_", " ").title()
                                st.markdown(f"**{cluster_name}**")
                                with open(nf, 'r', encoding='utf-8') as f:
                                    st.components.v1.html(f.read(), height=600, scrolling=True)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        st.exception(e)
