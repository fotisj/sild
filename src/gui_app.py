"""
Semantic Change Analysis - Streamlit GUI

This module provides the web interface for the semantic change analysis toolkit.
View logic is separated from business logic through small, focused functions.
"""
import streamlit as st
import os
import sys
import contextlib
import json
import glob
import time

# Ensure module path is available
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

CONFIG_FILE = "config.json"
OUTPUT_DIR = "output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Configuration Management
# =============================================================================

def get_default_config() -> dict:
    """Returns the default configuration dictionary."""
    return {
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
        "clustering_reduction": "None",
        "clustering_n_components": 50,
        "viz_reduction": "pca",
        "use_umap": True,
        "umap_n_components": 50,
        "n_samples": 50,
        "viz_max_instances": 100,
        "min_freq": 25,
        "layers": [-1],
        "layer_op": "mean",
        "lang": "en",
        "context_window": 0,
        "n_top_sentences": 10,
        "k_per_sentence": 6,
    }


def load_config() -> dict:
    """Loads configuration from file, merging with defaults."""
    default = get_default_config()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            loaded = json.load(f)
            for key, val in default.items():
                if key not in loaded:
                    loaded[key] = val
            return loaded
    return default


def save_config(config: dict) -> None:
    """Saves configuration to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


# =============================================================================
# Utility Functions
# =============================================================================

@contextlib.contextmanager
def capture_stdout(output_func):
    """Captures stdout and sends it to an output function."""
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


def get_db_paths(config: dict) -> tuple[str, str]:
    """Returns the paths to both corpus databases."""
    db_t1 = os.path.join(config["data_dir"], "corpus_t1.db")
    db_t2 = os.path.join(config["data_dir"], "corpus_t2.db")
    return db_t1, db_t2


def check_databases_exist(db_t1: str, db_t2: str) -> bool:
    """Checks if both corpus databases exist."""
    return os.path.exists(db_t1) and os.path.exists(db_t2)


def get_available_models() -> list[str]:
    """Gets list of models with pre-computed embeddings from ChromaDB."""
    try:
        from semantic_change.vector_store import VectorStore
        store = VectorStore(persistence_path="data/chroma_db")
        return store.get_available_models()
    except Exception:
        return []


@st.cache_resource
def get_embedder(model_name: str, layers=None, layer_op: str = "mean", lang: str = "en"):
    """Loads and caches the BERT Embedder model."""
    from semantic_change.embedding import BertEmbedder
    return BertEmbedder(model_name=model_name, layers=layers, layer_op=layer_op, lang=lang)


# =============================================================================
# Sidebar Components
# =============================================================================

def render_navigation() -> str:
    """Renders the navigation sidebar and returns the selected page."""
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        ["Analysis Dashboard", "Data Ingestion", "Embeddings & Models", "Corpus Reports"],
    )


def render_global_settings(config: dict) -> None:
    """Renders global settings in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Global Settings**")
    config["data_dir"] = st.sidebar.text_input(
        "Data Output Directory", value=config["data_dir"]
    )

    if st.sidebar.button("Save Configuration"):
        save_config(config)
        st.sidebar.success("Config saved!")


def render_period_labels(config: dict) -> tuple[str, str]:
    """Renders period label inputs and returns the labels."""
    st.sidebar.markdown("**Time Period Labels**")
    t1 = st.sidebar.text_input("Period 1 Label", value=config.get("period_t1_label", "1800"))
    t2 = st.sidebar.text_input("Period 2 Label", value=config.get("period_t2_label", "1900"))
    config["period_t1_label"] = t1
    config["period_t2_label"] = t2
    return t1, t2


def render_db_status(dbs_exist: bool) -> None:
    """Renders database status indicator."""
    if dbs_exist:
        st.sidebar.success("✅ Databases Ready")
    else:
        st.sidebar.warning("⚠️ Databases Missing")


# =============================================================================
# Data Ingestion Page
# =============================================================================

def render_ingestion_inputs(config: dict, period_t1_label: str, period_t2_label: str) -> tuple[str, str]:
    """Renders input directory configuration."""
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Corpus: {period_t1_label}")
        input_t1 = st.text_input(
            f"Input Directory ({period_t1_label})",
            value=config.get("input_dir_t1", "data_gutenberg/1800"),
        )
        config["input_dir_t1"] = input_t1

    with col2:
        st.subheader(f"Corpus: {period_t2_label}")
        input_t2 = st.text_input(
            f"Input Directory ({period_t2_label})",
            value=config.get("input_dir_t2", "data_gutenberg/1900"),
        )
        config["input_dir_t2"] = input_t2

    return input_t1, input_t2


def render_spacy_selector(config: dict) -> str:
    """Renders SpaCy model selector."""
    st.markdown("### Settings")
    spacy_model = st.selectbox(
        "SpaCy Model (Lemmatization)",
        ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
        index=0,
    )
    config["spacy_model"] = spacy_model
    return spacy_model


def run_ingestion_process(
    input_t1: str, input_t2: str,
    db_t1: str, db_t2: str,
    period_t1_label: str, period_t2_label: str,
    spacy_model: str
) -> None:
    """Executes the ingestion process with progress tracking."""
    log_container = st.empty()
    logs = []

    def update_log(text):
        logs.append(text)
        log_container.code("".join(logs[-30:]))

    start_time = time.time()
    pbar = st.progress(0)
    status = st.empty()

    def progress_callback(current, total, desc):
        if total > 0:
            pbar.progress(min(current / total, 1.0))
            status.text(f"{desc} ({current}/{total})")

    with st.spinner("Ingesting corpora... This may take a while."):
        try:
            from run_ingest import run_ingestion
            with capture_stdout(update_log):
                run_ingestion(
                    input_t1=input_t1,
                    input_t2=input_t2,
                    db_t1=db_t1,
                    db_t2=db_t2,
                    label_t1=period_t1_label,
                    label_t2=period_t2_label,
                    spacy_model=spacy_model,
                    progress_callback=progress_callback,
                )

            elapsed = time.time() - start_time
            st.success(f"Ingestion completed in {elapsed / 60:.2f} minutes!")
            st.balloons()
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            st.exception(e)


def render_danger_zone(db_t1: str, db_t2: str) -> None:
    """Renders the database deletion section."""
    st.markdown("---")
    st.subheader("🗑️ Danger Zone")
    st.warning("These actions will delete processed data.")

    if st.button("Wipe SQLite Databases", help="Deletes the corpus_t1.db and corpus_t2.db files."):
        from run_ingest import delete_databases
        success, message = delete_databases(db_t1, db_t2)
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(f"Failed to delete databases: {message}")


def render_ingestion_page(config: dict, db_t1: str, db_t2: str, period_t1_label: str, period_t2_label: str) -> None:
    """Renders the Data Ingestion page."""
    st.title("🗄️ Corpus Ingestion")
    st.markdown("""
    Process raw text files into a structured SQLite database.
    **This step must be done once before analysis.**
    """)

    input_t1, input_t2 = render_ingestion_inputs(config, period_t1_label, period_t2_label)
    spacy_model = render_spacy_selector(config)

    st.info(f"Output files will be saved to: `{db_t1}` and `{db_t2}`")

    if st.button("Run Ingestion Process", type="primary"):
        save_config(config)
        if not os.path.exists(input_t1) or not os.path.exists(input_t2):
            st.error("One or both input directories do not exist.")
        else:
            run_ingestion_process(input_t1, input_t2, db_t1, db_t2, period_t1_label, period_t2_label, spacy_model)

    render_danger_zone(db_t1, db_t2)


# =============================================================================
# Embeddings & Models Page
# =============================================================================

def render_model_selector(config: dict) -> str:
    """Renders the model selection UI and returns the selected model name."""
    common_models = [
        "bert-base-uncased",
        "roberta-base",
        "google/modernbert-base-uncased",
        "distilbert-base-uncased",
        "xlm-roberta-base",
        "Other (Custom)",
    ]

    current_model = config["model_name"]
    default_index = common_models.index(current_model) if current_model in common_models else common_models.index("Other (Custom)")

    selected = st.selectbox("Select Model", common_models, index=default_index)

    if selected == "Other (Custom)":
        return st.text_input("Hugging Face Model ID", value=current_model, help="e.g. bert-large-uncased")
    return selected


def render_layer_config(config: dict) -> None:
    """Renders layer selection and combination configuration."""
    with st.expander("Layer Selection & Combination"):
        layer_options = {
            "Last Layer (-1)": -1,
            "2nd Last Layer (-2)": -2,
            "3rd Last Layer (-3)": -3,
            "4th Last Layer (-4)": -4,
        }

        current_layers = config.get("layers", [-1])
        default_labels = [label for label, val in layer_options.items() if val in current_layers]

        selected_labels = st.multiselect("Layers to Use", list(layer_options.keys()), default=default_labels)
        config["layers"] = [layer_options[l] for l in selected_labels]

        combo_options = ["Mean", "Sum", "Concat"]
        current_op = config.get("layer_op", "mean").title()
        if current_op not in combo_options:
            current_op = "Mean"

        selected_op = st.selectbox("Layer Combination Method", combo_options, index=combo_options.index(current_op))
        config["layer_op"] = selected_op.lower()


def run_batch_embedding_process(
    db_t1: str, db_t2: str,
    model_name: str, min_freq: int,
    custom_words: list[str]
) -> None:
    """Executes the batch embedding generation process."""
    batch_log = st.empty()
    logs = []

    def update_log(text):
        if not text.startswith("\r") and "%" not in text:
            logs.append(text)
            batch_log.code("".join(logs[-20:]))

    pbar = st.progress(0)
    pbar_text = st.empty()

    def progress_callback(current, total, desc):
        if total > 0:
            progress = min(current / total, 1.0)
            pbar.progress(progress)
            pbar_text.text(f"{desc}: {current}/{total} ({int(progress * 100)}%)")

    with st.spinner(f"Generating embeddings for {model_name}..."):
        try:
            from semantic_change.embeddings_generation import run_batch_generation
            with capture_stdout(update_log):
                run_batch_generation(
                    db_path_t1=db_t1,
                    db_path_t2=db_t2,
                    model_name=model_name,
                    min_freq=min_freq,
                    additional_words=custom_words,
                    progress_callback=progress_callback,
                    reset_collections=True,
                )
            st.success("Batch Processing Complete!")
            pbar.progress(1.0)
        except Exception as e:
            st.error(f"Batch process failed: {e}")


def render_create_embeddings_tab(config: dict, db_t1: str, db_t2: str) -> None:
    """Renders the 'Create New Embeddings' tab content."""
    st.subheader("Generate Embeddings")
    st.info("This process uses the chosen model to compute vector representations for the words in your ingested corpus.")

    col1, col2 = st.columns(2)
    with col1:
        model_name = render_model_selector(config)
        config["model_name"] = model_name

        min_freq = st.number_input(
            "Minimum Frequency",
            min_value=5,
            value=config.get("min_freq", 25),
            step=5,
        )
        config["min_freq"] = min_freq

    with col2:
        custom_words_input = st.text_area(
            "Custom Words (optional)",
            placeholder="e.g. apple, banana\n(comma or newline separated)",
            help="Add specific words to the batch processing list, regardless of frequency.",
        )

    st.markdown("#### Advanced Model Config")
    render_layer_config(config)

    if st.button("Start Batch Process", type="primary"):
        save_config(config)
        custom_words = []
        if custom_words_input:
            raw_words = custom_words_input.replace("\n", ",").split(",")
            custom_words = [w.strip() for w in raw_words if w.strip()]

        run_batch_embedding_process(db_t1, db_t2, model_name, min_freq, custom_words)


def render_manage_embeddings_tab() -> None:
    """Renders the 'Manage Existing' embeddings tab content."""
    st.subheader("Existing Embedding Sets")
    available_models = get_available_models()

    if not available_models:
        st.info("No embeddings found in the database.")
        return

    st.write(f"Found {len(available_models)} processed models:")
    for m in available_models:
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"**{m}**")
        if col2.button("Delete", key=f"del_{m}"):
            delete_model_embeddings(m)


def delete_model_embeddings(model_safe_name: str) -> None:
    """Deletes embeddings for a specific model."""
    try:
        from semantic_change.vector_store import VectorStore
        store = VectorStore(persistence_path="data/chroma_db")
        store.delete_collection(f"embeddings_t1_{model_safe_name}")
        store.delete_collection(f"embeddings_t2_{model_safe_name}")
        st.success(f"Deleted embeddings for {model_safe_name}")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Delete failed: {e}")


def render_embeddings_page(config: dict, db_t1: str, db_t2: str) -> None:
    """Renders the Embeddings & Models page."""
    st.title("🧬 Embeddings & Models")
    st.markdown("""
    Manage the LLM models used for semantic analysis. You can generate embeddings for multiple models and switch between them in the Dashboard.
    """)

    tab_new, tab_manage = st.tabs(["Create New Embeddings", "Manage Existing"])

    with tab_new:
        render_create_embeddings_tab(config, db_t1, db_t2)

    with tab_manage:
        render_manage_embeddings_tab()


# =============================================================================
# Corpus Reports Page
# =============================================================================

def render_reports_page(config: dict, db_t1: str, db_t2: str, period_t1_label: str, period_t2_label: str, dbs_exist: bool) -> None:
    """Renders the Corpus Reports page."""
    st.title("📊 Corpus Statistics")

    if not dbs_exist:
        st.error("Please run Ingestion first.")
        return

    st.write("Compare word frequencies between the two time periods.")
    report_top_n = st.number_input("Top N Shared Words", value=30, min_value=10)

    available_models = get_available_models()
    include_semantic_change = bool(available_models)
    report_model_name = None

    if available_models:
        report_model_name = st.selectbox(
            "Select Model for Semantic Change",
            options=available_models,
            help="Select the embedding set to use for semantic change computation.",
        )
    else:
        st.warning("No embeddings found. Please go to 'Embeddings & Models' to generate them first.")

    if st.button("Generate Comparison Report"):
        generate_comparison_report_ui(
            db_t1, db_t2,
            period_t1_label, period_t2_label,
            report_top_n, report_model_name,
            include_semantic_change
        )

    display_existing_report()


def generate_comparison_report_ui(
    db_t1: str, db_t2: str,
    period_t1_label: str, period_t2_label: str,
    top_n: int, model_name: str | None,
    include_semantic_change: bool
) -> None:
    """Generates and displays the comparison report."""
    with st.spinner("Generating Report..."):
        try:
            from semantic_change.reporting import generate_comparison_report
            from semantic_change.corpus import CorpusManager

            manager = CorpusManager()
            manager.add_corpus(period_t1_label, os.path.dirname(db_t1), db_t1)
            manager.add_corpus(period_t2_label, os.path.dirname(db_t2), db_t2)

            report_path = os.path.join(OUTPUT_DIR, "processing_report.md")

            generate_comparison_report(
                manager.get_corpus(period_t1_label),
                manager.get_corpus(period_t2_label),
                top_n=top_n,
                output_path=report_path,
                model_name=model_name,
                include_semantic_change=include_semantic_change,
            )
            st.success(f"Report saved to {report_path}")
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)


def display_existing_report() -> None:
    """Displays an existing report if available."""
    report_path = os.path.join(OUTPUT_DIR, "processing_report.md")
    if os.path.exists(report_path):
        st.markdown("---")
        st.subheader("Existing Report")
        with open(report_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())


# =============================================================================
# Analysis Dashboard Page
# =============================================================================

def render_dashboard_parameters(config: dict, available_models: list[str]) -> dict:
    """Renders the parameter inputs for the dashboard and returns selected values."""
    params = {}

    # Model selection
    selected_safe_model = st.selectbox(
        "Select Embedding Set",
        available_models,
        index=0 if available_models else None,
        help="Select the pre-computed embeddings to use.",
    )
    params["selected_safe_model"] = selected_safe_model

    st.info(f"Using embeddings from: {selected_safe_model}")
    params["model_name"] = st.text_input(
        "Confirm Model ID (for Tokenizer)",
        value=config["model_name"],
    )

    # Warn about mismatch
    if params["model_name"].replace("/", "_").replace("-", "_") != selected_safe_model:
        st.warning("⚠️ Model ID does not match the selected embedding set! Neighbor fallback might fail.")

    params["target_word"] = st.text_input("Target Word", value=config["target_word"])

    # POS filter
    pos_options = ["None", "NOUN", "VERB", "ADJ"]
    current_pos = config.get("pos_filter", "None")
    if current_pos not in pos_options:
        current_pos = "None"
    params["pos_filter"] = st.selectbox("POS Filter", options=pos_options, index=pos_options.index(current_pos))

    params["n_samples"] = st.number_input(
        "Samples per Period",
        min_value=10,
        max_value=2000,
        value=config["n_samples"],
    )

    return params


def render_clustering_parameters(config: dict) -> dict:
    """Renders clustering algorithm parameters."""
    params = {}

    wsi_options = ["hdbscan", "kmeans", "spectral", "agglomerative"]
    params["wsi_algorithm"] = st.selectbox(
        "Clustering Algorithm",
        options=wsi_options,
        index=wsi_options.index(config["wsi_algorithm"]),
    )

    if params["wsi_algorithm"] == "hdbscan":
        params["min_cluster_size"] = st.number_input(
            "Min Cluster Size",
            min_value=2,
            value=config["min_cluster_size"],
        )
        params["n_clusters"] = config["n_clusters"]
    else:
        params["n_clusters"] = st.number_input(
            "Number of Clusters (k)",
            min_value=2,
            value=config["n_clusters"],
        )
        params["min_cluster_size"] = config["min_cluster_size"]

    return params


def render_reduction_parameters(config: dict) -> dict:
    """Renders dimensionality reduction parameters."""
    params = {}
    st.markdown("##### Dimensionality Reduction")

    # Pre-clustering reduction
    clust_options = ["None", "pca", "umap", "tsne"]
    current_clust = config.get("clustering_reduction", "None")
    if current_clust not in clust_options:
        current_clust = "None"

    params["clustering_reduction"] = st.selectbox(
        "Pre-clustering Reduction",
        options=clust_options,
        index=clust_options.index(current_clust),
        help="Reduce dimensions before clustering. Helps with high-dim embeddings (~768d).",
    )

    if params["clustering_reduction"] != "None":
        params["clustering_n_components"] = st.number_input(
            "Clustering Reduction Dims",
            min_value=2,
            max_value=200,
            value=config.get("clustering_n_components", 50),
        )
    else:
        params["clustering_n_components"] = config.get("clustering_n_components", 50)

    # Visualization reduction
    viz_options = ["pca", "umap", "tsne"]
    current_viz = config.get("viz_reduction", "pca")
    if current_viz not in viz_options:
        current_viz = "pca"

    params["viz_reduction"] = st.selectbox(
        "Visualization Reduction",
        options=viz_options,
        index=viz_options.index(current_viz),
        help="Method to reduce to 2D for plotting.",
    )

    return params


def render_neighbor_parameters(config: dict) -> dict:
    """Renders semantic neighbor parameters."""
    params = {}

    params["k_neighbors"] = st.number_input(
        "Neighbors (k)",
        min_value=1,
        value=config["k_neighbors"],
    )

    params["context_window"] = st.slider(
        "Context Window (chars around word)",
        min_value=0,
        max_value=5000,
        value=config.get("context_window", 0),
        help="0 means use the sentence from DB. >0 reads raw file around the word.",
    )

    st.markdown("##### Contextual Neighbors (MLM Aggregation)")
    params["n_top_sentences"] = st.number_input(
        "Sentences to sample",
        min_value=1,
        max_value=50,
        value=config.get("n_top_sentences", 10),
    )
    params["k_per_sentence"] = st.number_input(
        "Predictions per sentence",
        min_value=1,
        max_value=20,
        value=config.get("k_per_sentence", 6),
    )

    return params


def update_config_from_params(config: dict, params: dict) -> None:
    """Updates the config dict with values from params."""
    config["target_word"] = params.get("target_word", config["target_word"])
    config["pos_filter"] = params.get("pos_filter", config["pos_filter"])
    config["n_samples"] = params.get("n_samples", config["n_samples"])
    config["wsi_algorithm"] = params.get("wsi_algorithm", config["wsi_algorithm"])
    config["min_cluster_size"] = params.get("min_cluster_size", config["min_cluster_size"])
    config["n_clusters"] = params.get("n_clusters", config["n_clusters"])
    config["clustering_reduction"] = params.get("clustering_reduction", config["clustering_reduction"])
    config["clustering_n_components"] = params.get("clustering_n_components", config["clustering_n_components"])
    config["viz_reduction"] = params.get("viz_reduction", config["viz_reduction"])
    config["k_neighbors"] = params.get("k_neighbors", config["k_neighbors"])
    config["context_window"] = params.get("context_window", config["context_window"])
    config["n_top_sentences"] = params.get("n_top_sentences", config["n_top_sentences"])
    config["k_per_sentence"] = params.get("k_per_sentence", config["k_per_sentence"])
    config["model_name"] = params.get("model_name", config["model_name"])


def run_analysis(config: dict, params: dict, db_t1: str, db_t2: str, period_t1_label: str, period_t2_label: str) -> None:
    """Runs the semantic analysis with the given parameters."""
    logs = []
    log_area = st.empty()

    def update_logs(text):
        logs.append(text)
        log_area.code("".join(logs))

    with st.spinner("Analyzing..."):
        try:
            from main import run_single_analysis

            clust_red = params["clustering_reduction"] if params["clustering_reduction"] != "None" else None
            pos_filter = None if params["pos_filter"] == "None" else params["pos_filter"]

            with capture_stdout(update_logs):
                run_single_analysis(
                    target_word=params["target_word"],
                    db_path_t1=db_t1,
                    db_path_t2=db_t2,
                    period_t1_label=period_t1_label,
                    period_t2_label=period_t2_label,
                    model_name=params["model_name"],
                    k_neighbors=params["k_neighbors"],
                    min_cluster_size=params["min_cluster_size"],
                    n_clusters=params["n_clusters"],
                    wsi_algorithm=params["wsi_algorithm"],
                    pos_filter=pos_filter,
                    clustering_reduction=clust_red,
                    clustering_n_components=params["clustering_n_components"],
                    viz_reduction=params["viz_reduction"],
                    n_samples=params["n_samples"],
                    viz_max_instances=config["viz_max_instances"],
                    embedder=None,
                    context_window=params["context_window"],
                    n_top_sentences=params["n_top_sentences"],
                    k_per_sentence=params["k_per_sentence"],
                )

            st.success("Analysis Complete!")
            display_visualizations()

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)


def display_visualizations() -> None:
    """Displays the generated visualization HTML files."""
    st.subheader("Visualizations")

    viz_files = [
        ("time_period.html", "⏳ Time Period Clustering"),
        ("sense_clusters.html", "🧩 Sense Clusters"),
        ("sense_time_combined.html", "🎨 Sense × Time (Combined)"),
    ]

    for filename, title in viz_files:
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            st.markdown(f"### {title}")
            with open(path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)

    # Neighbor files
    neighbor_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "neighbors_cluster_*.html")))
    if neighbor_files:
        st.markdown("### 🕸️ Semantic Neighbors")
        for nf in neighbor_files:
            cluster_name = (
                os.path.basename(nf)
                .replace("neighbors_", "")
                .replace(".html", "")
                .replace("_", " ")
                .title()
            )
            st.markdown(f"**{cluster_name}**")
            with open(nf, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)


def render_dashboard_page(
    config: dict,
    db_t1: str, db_t2: str,
    period_t1_label: str, period_t2_label: str,
    dbs_exist: bool
) -> None:
    """Renders the Analysis Dashboard page."""
    st.title("🔎 Semantic Analysis")

    if not dbs_exist:
        st.error("Databases missing. Please go to 'Data Ingestion' first.")
        return

    available_models = get_available_models()
    if not available_models:
        st.warning("⚠️ No embeddings found. Please go to the **Embeddings & Models** tab to generate them first.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")
        params = render_dashboard_parameters(config, available_models)
        params.update(render_clustering_parameters(config))
        params.update(render_reduction_parameters(config))
        params.update(render_neighbor_parameters(config))

        update_config_from_params(config, params)
        run_btn = st.button("Run Analysis", type="primary")

    with col2:
        if run_btn:
            save_config(config)
            run_analysis(config, params, db_t1, db_t2, period_t1_label, period_t2_label)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(page_title="Semantic Change Analysis", layout="wide", page_icon="📚")

    # Initialize config in session state
    if "config" not in st.session_state:
        st.session_state.config = load_config()

    config = st.session_state.config

    # Sidebar
    page = render_navigation()
    render_global_settings(config)
    period_t1_label, period_t2_label = render_period_labels(config)

    # Database paths and status
    db_t1, db_t2 = get_db_paths(config)
    dbs_exist = check_databases_exist(db_t1, db_t2)
    render_db_status(dbs_exist)

    # Page routing
    if page == "Data Ingestion":
        render_ingestion_page(config, db_t1, db_t2, period_t1_label, period_t2_label)
    elif page == "Embeddings & Models":
        render_embeddings_page(config, db_t1, db_t2)
    elif page == "Corpus Reports":
        render_reports_page(config, db_t1, db_t2, period_t1_label, period_t2_label, dbs_exist)
    elif page == "Analysis Dashboard":
        render_dashboard_page(config, db_t1, db_t2, period_t1_label, period_t2_label, dbs_exist)


# Run the app
main()
