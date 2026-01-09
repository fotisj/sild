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
            # Merge with defaults to ensure all keys exist
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
    ["Analysis Dashboard", "Data Ingestion", "Embeddings Config", "Corpus Reports"]
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
            with st.spinner("Ingesting corpora... This may take a while."):
                try:
                    from semantic_change.ingestor import Ingestor
                    from semantic_change.corpus import Corpus
                    from semantic_change.reporting import generate_comparison_report

                    with capture_stdout(update_ingest_log):
                        ingestor = Ingestor(model=spacy_model)

                        print(f"--- Processing {period_t1_label} from {input_t1} ---")
                        ingestor.preprocess_corpus(input_t1, db_t1)

                        print(f"--- Processing {period_t2_label} from {input_t2} ---")
                        ingestor.preprocess_corpus(input_t2, db_t2)

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


# --- Page: Embeddings Config ---
elif page == "Embeddings Config":
    st.title("🧬 Embedding Model Configuration")
    
    st.markdown("Configure the Transformer model used to represent word meanings.")
    
    model_name = st.text_input("Hugging Face Model Name", value=st.session_state.config["model_name"])
    st.session_state.config["model_name"] = model_name
    
    st.markdown("### Advanced Settings")
    st.info("Configure which transformer layers to use for generating embeddings.")
    
    # Layer Selection
    layer_options = {
        "Last Layer (-1)": -1,
        "2nd Last Layer (-2)": -2,
        "3rd Last Layer (-3)": -3,
        "4th Last Layer (-4)": -4
    }
    
    # Map current config to labels
    default_layers = []
    current_layers = st.session_state.config.get("layers", [-1])
    for label, val in layer_options.items():
        if val in current_layers:
            default_layers.append(label)
            
    selected_labels = st.multiselect(
        "Layers to Use", 
        list(layer_options.keys()), 
        default=default_layers
    )
    
    # Update config based on selection
    st.session_state.config["layers"] = [layer_options[l] for l in selected_labels]
    
    # Combination Method
    combo_options = ["Mean", "Sum", "Concat"]
    current_op = st.session_state.config.get("layer_op", "mean").title()
    if current_op not in combo_options: current_op = "Mean"
    
    selected_op = st.selectbox(
        "Layer Combination Method",
        combo_options,
        index=combo_options.index(current_op)
    )
    st.session_state.config["layer_op"] = selected_op.lower()
    
    lang_code = st.text_input("Language Code (for alignment)", value=st.session_state.config.get("lang", "en"))
    st.session_state.config["lang"] = lang_code

    if st.button("Load & Verify Model"):
        with st.spinner(f"Loading {model_name}..."):
            try:
                embedder = get_embedder(
                    model_name, 
                    layers=st.session_state.config["layers"],
                    layer_op=st.session_state.config["layer_op"],
                    lang=st.session_state.config["lang"]
                )
                st.success(f"Successfully loaded '{model_name}'!")
                st.write(f"Vocab Size: {embedder.tokenizer.vocab_size}")
                st.write(f"Hidden Dimension: {embedder.mlm_model.config.hidden_size}")
                st.write(f"Configured to use layers {st.session_state.config['layers']} combined via {st.session_state.config['layer_op']}")
            except Exception as e:
                st.error(f"Failed to load model: {e}")


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
                    
        # Check for existing report
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
        tab_single, tab_batch = st.tabs(["Single Word Analysis", "Batch Analysis"])
        
        # --- Single Word Tab ---
        with tab_single:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Parameters")
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
                        
                        # Convert "None" string to actual None
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
                                context_window=context_window
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

        # --- Batch Analysis Tab ---
        with tab_batch:
            st.header("Batch Analysis")
            st.write("Process all shared nouns with a minimum frequency to pre-compute embeddings.")
            
            min_freq_val = st.number_input("Minimum Frequency", min_value=5, value=st.session_state.config.get("min_freq", 25), step=5)
            st.session_state.config["min_freq"] = min_freq_val
            
            custom_words_input = st.text_area("Custom Words (comma or newline separated)", 
                                              placeholder="e.g. apple, banana, car\nor\napple\nbanana",
                                              help="Add specific words to the batch processing list, regardless of their frequency rank.")
            
            if st.button("Start Batch Process"):
                save_config(st.session_state.config)
                
                # Parse custom words
                custom_words = []
                if custom_words_input:
                    # Replace newlines with commas and split
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

                with st.spinner("Processing..."):
                    try:
                        from run_batch_analysis import run_batch_process
                        
                        # Generate safe labels for embedding folders
                        safe_label_t1 = period_t1_label.replace(" ", "_").replace("/", "_")
                        safe_label_t2 = period_t2_label.replace(" ", "_").replace("/", "_")

                        with capture_stdout(update_batch_logs):
                            run_batch_process(
                                db_path_1800=db_t1,
                                db_path_1900=db_t2,
                                output_dir_1800=os.path.join(st.session_state.config["data_dir"], f"embeddings/{safe_label_t1}"),
                                output_dir_1900=os.path.join(st.session_state.config["data_dir"], f"embeddings/{safe_label_t2}"),
                                model_name=st.session_state.config["model_name"],
                                min_freq=min_freq_val,
                                additional_words=custom_words,
                                progress_callback=progress_callback
                            )
                        st.success("Batch Processing Complete!")
                        pbar.progress(1.0)
                    except Exception as e:
                        st.error(f"Batch process failed: {e}")

            st.markdown("---")
            st.subheader("Semantic Change Ranking")
            st.write("Compute the semantic shift (cosine distance) for all words in the database.")
            
            if st.button("Calculate Semantic Change"):
                rank_pbar = st.progress(0)
                rank_status = st.empty()
                
                def rank_progress_callback(current, total, desc):
                    if total > 0:
                        progress = min(current / total, 1.0)
                        rank_pbar.progress(progress)
                        rank_status.text(f"{desc}: {current}/{total} ({int(progress*100)}%)")

                with st.spinner("Calculating distances..."):
                    try:
                        from rank_semantic_change import compute_centroid_shift
                        import pandas as pd
                        
                        output_csv = os.path.join(OUTPUT_DIR, "semantic_change_ranking.csv")
                        
                        # Use batch log area for output
                        batch_log_rank = st.empty()
                        def update_rank_logs(text):
                            batch_log_rank.code(text)

                        with capture_stdout(update_rank_logs):
                             compute_centroid_shift(
                                 min_freq=min_freq_val, 
                                 output_file=output_csv, 
                                 progress_callback=rank_progress_callback,
                                 model_name=st.session_state.config["model_name"]
                             )
                        
                        rank_pbar.progress(1.0)
                        if os.path.exists(output_csv):
                            st.success(f"Ranking saved to {output_csv}")
                            df = pd.read_csv(output_csv)
                            st.dataframe(df.head(20))
                        else:
                            st.warning("No results generated. Check if the database is populated.")
                            
                    except Exception as e:
                        st.error(f"Ranking failed: {e}")