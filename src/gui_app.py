"""
Semantic Change Analysis - Streamlit GUI

This module provides the web interface for the semantic change analysis toolkit.
View logic is separated from business logic through small, focused functions.
"""
import warnings
import logging
import threading
import sys

# Suppress Streamlit's ScriptRunContext warning (harmless, occurs during background processing)
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Thread.*ScriptRunContext.*")

# Suppress streamlit logger warnings about missing context
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Suppress threading excepthook warnings (these occur when stqdm updates from worker threads)
_original_excepthook = threading.excepthook
def _silent_excepthook(args):
    if "ScriptRunContext" in str(args.exc_value):
        return  # Silently ignore ScriptRunContext warnings
    _original_excepthook(args)
threading.excepthook = _silent_excepthook

# Suppress sys.unraisablehook warnings
_original_unraisablehook = sys.unraisablehook
def _silent_unraisablehook(unraisable):
    if "ScriptRunContext" in str(unraisable.exc_value):
        return  # Silently ignore ScriptRunContext warnings
    _original_unraisablehook(unraisable)
sys.unraisablehook = _silent_unraisablehook

import streamlit as st
import os
import contextlib
import json
import glob
import time
import traceback
import zipfile
from datetime import datetime
from stqdm import stqdm

# Ensure module path is available
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import refactored modules
from semantic_change.config_manager import (
    load_config,
    save_config,
    get_db_paths,
    check_databases_exist,
    ensure_project_id,
)
from utils.dependencies import check_spacy_transformer_deps
from semantic_change.services import StatsService, ClusterService

OUTPUT_DIR = "output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Utility Functions
# =============================================================================

@contextlib.contextmanager
def capture_output(output_func):
    """Captures stdout and stderr, sends them to an output function."""
    class StreamlitWriter:
        def write(self, text):
            output_func(text)
        def flush(self):
            pass

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    writer = StreamlitWriter()
    sys.stdout = writer
    sys.stderr = writer
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@st.cache_resource
def get_vector_store():
    """Gets a cached VectorStore instance to avoid multiple ChromaDB connections."""
    from semantic_change.vector_store import VectorStore
    return VectorStore(persistence_path="data/chroma_db")


def clear_vector_store_cache():
    """Clears the cached VectorStore instance (call after delete operations)."""
    get_vector_store.clear()


def get_available_models(project_id: str) -> list[str]:
    """Gets list of models with pre-computed embeddings from ChromaDB for a project."""
    if not project_id:
        return []
    try:
        store = get_vector_store()
        return store.get_available_models(project_id)
    except Exception:
        return []


@st.cache_resource
def get_embedder(model_name: str, layers=None, layer_op: str = "mean", lang: str = "en", filter_model: str = None):
    """Loads and caches the BERT Embedder model."""
    from semantic_change.embedding import BertEmbedder
    return BertEmbedder(model_name=model_name, layers=layers, layer_op=layer_op, lang=lang, filter_model=filter_model)


# =============================================================================
# Sidebar Components
# =============================================================================

def render_navigation() -> str:
    """Renders the navigation sidebar and returns the selected page."""
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        ["SCA Macro", "SCA Micro", "Data Ingestion", "Embeddings & Models", "Corpus Reports"],
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


def render_max_files_selector(config: dict) -> int | None:
    """Renders max files selector for testing."""
    use_limit = st.checkbox(
        "Limit files (for testing)",
        value=config.get("max_files") is not None,
        help="Process only a random subset of files for faster testing"
    )

    if use_limit:
        max_files = st.number_input(
            "Max files per period",
            min_value=1,
            max_value=1000,
            value=config.get("max_files") or 25,
            step=5,
        )
        config["max_files"] = max_files
        return max_files
    else:
        config["max_files"] = None
        return None


def render_encoding_selector(config: dict) -> str:
    """Renders file encoding selector."""
    encodings = [
        ("UTF-8", "utf-8"),
        ("Windows/ANSI (CP1252)", "cp1252"),
        ("Latin-1 (ISO-8859-1)", "latin-1"),
        ("Other", "other"),
    ]
    encoding_labels = [e[0] for e in encodings]
    encoding_values = [e[1] for e in encodings]

    current_encoding = config.get("file_encoding", "utf-8")
    if current_encoding in encoding_values:
        default_index = encoding_values.index(current_encoding)
    else:
        default_index = encoding_labels.index("Other")

    selected_label = st.selectbox("File Encoding", encoding_labels, index=default_index)
    selected_value = encoding_values[encoding_labels.index(selected_label)]

    if selected_value == "other":
        file_encoding = st.text_input(
            "Custom Encoding",
            value=current_encoding if current_encoding not in encoding_values else "",
            help="e.g., cp850, iso-8859-15, utf-16",
        )
    else:
        file_encoding = selected_value

    config["file_encoding"] = file_encoding
    return file_encoding


def render_spacy_selector(config: dict) -> str:
    """Renders SpaCy model selector with custom model support."""
    st.markdown("### Settings")
    common_models = [
        "en_core_web_sm",
        "en_core_web_md",
        "en_core_web_lg",
        "en_core_web_trf",
        "Other (Custom)",
    ]

    current_model = config.get("spacy_model", "en_core_web_sm")
    default_index = common_models.index(current_model) if current_model in common_models[:-1] else common_models.index("Other (Custom)")

    selected = st.selectbox("SpaCy Model (Lemmatization)", common_models, index=default_index)

    if selected == "Other (Custom)":
        spacy_model = st.text_input(
            "SpaCy Model Name",
            value=current_model if current_model not in common_models[:-1] else "",
            help="e.g. de_core_news_sm, es_core_news_md, or any spaCy model name",
        )
    else:
        spacy_model = selected

    # Show warning for transformer models
    if spacy_model.endswith("_trf"):
        st.info("Transformer models require additional dependencies. They will be installed automatically if needed.")

    config["spacy_model"] = spacy_model
    return spacy_model


def run_ingestion_process(
    input_t1: str, input_t2: str,
    db_t1: str, db_t2: str,
    period_t1_label: str, period_t2_label: str,
    spacy_model: str,
    file_encoding: str = "utf-8",
    max_files: int | None = None
) -> None:
    """Executes the ingestion process with progress tracking."""
    log_container = st.empty()
    logs = []

    def update_log(text):
        logs.append(text)
        log_container.code("".join(logs[-30:]))

    start_time = time.time()

    with st.spinner("Ingesting corpora... This may take a while."):
        try:
            from run_ingest import run_ingestion
            with capture_output(update_log):
                run_ingestion(
                    input_t1=input_t1,
                    input_t2=input_t2,
                    db_t1=db_t1,
                    db_t2=db_t2,
                    label_t1=period_t1_label,
                    label_t2=period_t2_label,
                    spacy_model=spacy_model,
                    file_encoding=file_encoding,
                    max_files=max_files,
                    tqdm_class=stqdm,
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

    # Display feedback from previous deletion (persisted across rerun)
    if "db_delete_message" in st.session_state:
        msg_type, msg_text = st.session_state.db_delete_message
        if msg_type == "success":
            st.success(msg_text)
        else:
            st.error(msg_text)
        del st.session_state.db_delete_message

    if st.button("Wipe SQLite Databases", help="Deletes the corpus_t1.db and corpus_t2.db files."):
        from run_ingest import delete_databases
        success, message = delete_databases(db_t1, db_t2)
        if success:
            st.session_state.db_delete_message = ("success", message)
        else:
            st.session_state.db_delete_message = ("error", f"Failed to delete databases: {message}")
        st.rerun()


def render_ingestion_page(config: dict, db_t1: str, db_t2: str, period_t1_label: str, period_t2_label: str) -> None:
    """Renders the Data Ingestion page."""
    st.title("🗄️ Corpus Ingestion")
    st.markdown("""
    Process raw text files into a structured SQLite database.
    **This step must be done once before analysis.**
    """)

    input_t1, input_t2 = render_ingestion_inputs(config, period_t1_label, period_t2_label)
    file_encoding = render_encoding_selector(config)
    max_files = render_max_files_selector(config)
    spacy_model = render_spacy_selector(config)

    st.info(f"Output files will be saved to: `{db_t1}` and `{db_t2}`")

    # Check if restart is needed (from previous dependency installation)
    if st.session_state.get("spacy_restart_needed"):
        st.warning("Dependencies were installed. Please restart the app to use transformer models.")
        if st.button("Restart App", type="primary"):
            st.session_state.spacy_restart_needed = False
            st.rerun()
        return

    if st.button("Run Ingestion Process", type="primary"):
        save_config(config)
        if not os.path.exists(input_t1) or not os.path.exists(input_t2):
            st.error("One or both input directories do not exist.")
            return

        # Check transformer dependencies before running
        with st.spinner("Checking dependencies..."):
            ready, message = check_spacy_transformer_deps(spacy_model)

        if not ready:
            if "restart" in message.lower():
                st.session_state.spacy_restart_needed = True
                st.warning(message)
                st.rerun()
            else:
                st.error(message)
            return

        run_ingestion_process(input_t1, input_t2, db_t1, db_t2, period_t1_label, period_t2_label, spacy_model, file_encoding, max_files)

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
    # Use common_models[:-1] to exclude "Other (Custom)" from the check
    default_index = common_models.index(current_model) if current_model in common_models[:-1] else common_models.index("Other (Custom)")

    selected = st.selectbox("Select Model", common_models, index=default_index, key="embedding_model_selector")

    if selected == "Other (Custom)":
        # Pre-fill with current model if it's a custom one, otherwise empty
        prefill = current_model if current_model not in common_models[:-1] else ""
        return st.text_input("Hugging Face Model ID", value=prefill, help="e.g. bert-large-uncased")
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

        combo_options = ["Mean", "Median", "Sum", "Concat"]
        current_op = config.get("layer_op", "mean").title()
        if current_op not in combo_options:
            current_op = "Mean"

        selected_op = st.selectbox("Layer Combination Method", combo_options, index=combo_options.index(current_op))
        config["layer_op"] = selected_op.lower()


def run_batch_embedding_process(
    project_id: str,
    db_t1: str, db_t2: str,
    model_name: str, min_freq: int,
    test_mode: bool = False,
    max_samples: int = 200,
    pooling_strategy: str = "mean",
    layers: list[int] = None,
    layer_op: str = "mean",
    staged: bool = False,
    resume: bool = False
) -> None:
    """Executes the batch embedding generation process."""
    if layers is None:
        layers = [-1]
    layers_str = ",".join(str(l) for l in layers)
    action_str = "Resuming" if resume else "Starting"

    # Use st.status for better progress visibility during long operations
    with st.status(f"{action_str} embedding generation...", expanded=True) as status:
        st.write(f"**Model:** {model_name}")
        st.write(f"**Pooling:** {pooling_strategy}, **Layers:** [{layers_str}], **Layer op:** {layer_op}")

        log_placeholder = st.empty()
        logs = []

        def update_log(text):
            if text.strip():  # Only update for non-empty text
                logs.append(text)
                # Keep only last 50 lines
                log_placeholder.code("".join(logs[-50:]), language=None)

        try:
            from semantic_change.embeddings_generation import run_batch_generation

            with capture_output(update_log):
                run_batch_generation(
                    project_id=project_id,
                    db_path_t1=db_t1,
                    db_path_t2=db_t2,
                    model_name=model_name,
                    min_freq=min_freq,
                    max_samples=max_samples,
                    reset_collections=not resume,  # Do not reset if resuming
                    test_mode=test_mode,
                    tqdm_class=stqdm,
                    pooling_strategy=pooling_strategy,
                    layers=layers,
                    layer_op=layer_op,
                    staged=staged,
                    resume=resume
                )
            status.update(label="Batch Processing Complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label=f"Batch process failed: {e}", state="error", expanded=True)
            st.code(traceback.format_exc(), language="python")


def render_create_embeddings_tab(config: dict, db_t1: str, db_t2: str) -> None:
    """Renders the 'Create New Embeddings' tab content."""
    st.subheader("Generate Embeddings")
    st.info("This process uses the chosen model to compute vector representations for the words in your ingested corpus.")

    model_name = render_model_selector(config)
    config["model_name"] = model_name

    col1, col2 = st.columns(2)
    with col1:
        min_freq = st.number_input(
            "Minimum Frequency",
            min_value=5,
            value=config.get("min_freq", 25),
            step=5,
        )
        config["min_freq"] = min_freq

    with col2:
        max_samples = st.number_input(
            "Max Samples per Word",
            min_value=10,
            value=config.get("batch_max_samples", 200),
            step=50,
            help="Maximum number of sentence samples to collect per word per period."
        )
        config["batch_max_samples"] = max_samples

    # Test mode option
    test_mode = st.checkbox(
        "Test mode (quick)",
        value=False,
        help="For testing: 25 shared nouns, min freq 50, 50 embeddings per word per period"
    )

    # Staged mode option for faster processing
    staged_mode = st.checkbox(
        "Staged mode (faster)",
        value=True,
        help="Write embeddings to NPZ files first, then bulk import to ChromaDB. "
             "Significantly faster for large batches due to reduced I/O overhead. "
             "Also enables resume capability if interrupted."
    )

    st.markdown("#### Advanced Model Config")
    render_layer_config(config)

    # Pooling strategy selection
    st.markdown("#### Subword Pooling Strategy")
    pooling_options = {
        "mean": "Mean (default) - Average all subword tokens",
        "first": "First - Use only the first subword token",
        "lemma_aligned": "Lemma-aligned - Pool only subwords matching lemma tokenization",
        "weighted": "Weighted - Position-weighted (earlier tokens get higher weight)",
        "lemma_replacement": "TokLem - Replace target with lemma before embedding (Laicher et al. 2021)",
    }
    pooling_strategy = st.selectbox(
        "Pooling Strategy",
        options=list(pooling_options.keys()),
        format_func=lambda x: pooling_options[x],
        index=list(pooling_options.keys()).index(config.get("pooling_strategy", "mean")),
        help="How to aggregate subword token embeddings into a single word embedding. "
             "'TokLem' is recommended for reducing morphological bias in semantic clustering."
    )
    config["pooling_strategy"] = pooling_strategy

    col_start, col_resume = st.columns([1, 1])
    
    with col_start:
        if st.button("Start Batch Process", type="primary"):
            save_config(config)
            run_batch_embedding_process(
                config["project_id"], db_t1, db_t2, model_name, min_freq,
                test_mode=test_mode, max_samples=max_samples, pooling_strategy=pooling_strategy,
                layers=config.get("layers", [-1]), layer_op=config.get("layer_op", "mean"),
                staged=staged_mode,
                resume=False
            )

    with col_resume:
        # Only show resume if staged mode is enabled (as resume depends on staging)
        if staged_mode:
            if st.button("Resume Batch Process", help="Resume interrupted processing from staged files."):
                save_config(config)
                run_batch_embedding_process(
                    config["project_id"], db_t1, db_t2, model_name, min_freq,
                    test_mode=test_mode, max_samples=max_samples, pooling_strategy=pooling_strategy,
                    layers=config.get("layers", [-1]), layer_op=config.get("layer_op", "mean"),
                    staged=staged_mode,
                    resume=True
                )


def render_manage_embeddings_tab(project_id: str) -> None:
    """Renders the 'Manage Existing' embeddings tab content."""
    st.subheader("Existing Embedding Sets")

    # Check if we're in the middle of a delete operation
    if st.session_state.get("deleting_model"):
        model_to_delete = st.session_state["deleting_model"]
        with st.status(f"Deleting {model_to_delete}...", expanded=True) as status:
            try:
                # Clear cache first to release any locks
                st.write("Clearing vector store cache...")
                clear_vector_store_cache()

                st.write("Connecting to ChromaDB...")
                store = get_vector_store()

                # Use the new delete_model_embeddings method
                st.write("Deleting embeddings... (this may take a while for large collections)")
                success, message, count_t1, count_t2 = store.delete_model_embeddings(
                    project_id, model_to_delete
                )

                if count_t1 > 0 or count_t2 > 0:
                    st.write(f"Found {count_t1} + {count_t2} = {count_t1 + count_t2} embeddings")
                st.write(f"Result: {message}")

                # Clear cache again after deletion to ensure fresh state
                clear_vector_store_cache()

                if success:
                    status.update(label=f"Deleted {model_to_delete}", state="complete")
                else:
                    status.update(label=f"No embeddings found for {model_to_delete}", state="complete")
            except Exception as e:
                status.update(label=f"Delete failed: {e}", state="error")
                st.code(traceback.format_exc())
            finally:
                st.session_state["deleting_model"] = None
                time.sleep(1)
                st.rerun()
        return

    available_models = get_available_models(project_id)

    if not available_models:
        st.info("No embeddings found in the database.")
        return

    st.write(f"Found {len(available_models)} processed models:")
    for m in available_models:
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"**{m}**")
        if col2.button("Delete", key=f"del_{m}"):
            st.session_state["deleting_model"] = m
            st.rerun()


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
        render_manage_embeddings_tab(config["project_id"])


# =============================================================================
# Corpus Reports Page
# =============================================================================

def render_db_stats_summary(config: dict, db_t1: str, db_t2: str, period_t1_label: str, period_t2_label: str) -> None:
    """Displays a summary of database statistics using StatsService."""
    stats_service = StatsService()

    # SQLite stats
    st.markdown("#### Database Status")

    stats1 = stats_service.get_corpus_stats(db_t1, period_t1_label)
    stats2 = stats_service.get_corpus_stats(db_t2, period_t2_label)

    if stats1 and stats2:
        total_docs = stats1.files + stats2.files
        total_sents = stats1.sentences + stats2.sentences
        total_tokens = stats1.tokens + stats2.tokens

        st.code(
            f"SQLite:  {total_docs:,} documents, {total_sents:,} sentences, {total_tokens:,} tokens\n"
            f"  {period_t1_label}:     {stats1.files:,} documents, {stats1.sentences:,} sentences, {stats1.tokens:,} tokens\n"
            f"  {period_t2_label}:     {stats2.files:,} documents, {stats2.sentences:,} sentences, {stats2.tokens:,} tokens",
            language=None
        )
    else:
        st.warning("Could not read SQLite stats")

    # ChromaDB stats
    try:
        from semantic_change.vector_store import VectorStore
        store = VectorStore(persistence_path="data/chroma_db")
        project_id = config.get("project_id", "")
        available_models = store.get_available_models(project_id) if project_id else []

        if available_models:
            # Get stats for first available model using StatsService
            model = available_models[0]
            embedding_stats = stats_service.get_embedding_stats(project_id, model)

            if embedding_stats:
                threshold = config.get("min_freq", "N/A")
                st.code(
                    f"Chroma ({model}):\n"
                    f"  Total:           {embedding_stats.total_embeddings:,} embeddings\n"
                    f"  Unique Words:    {embedding_stats.unique_lemmas}\n"
                    f"  Freq. Threshold: {threshold}",
                    language=None
                )
            else:
                st.code(f"Chroma ({model}): Could not read stats", language=None)
        else:
            st.code("Chroma: No embeddings found", language=None)
    except Exception as e:
        st.code(f"Chroma: Not available ({e})", language=None)

    st.markdown("---")


def render_reports_page(config: dict, db_t1: str, db_t2: str, period_t1_label: str, period_t2_label: str, dbs_exist: bool) -> None:
    """Renders the Corpus Reports page."""
    st.title("📊 Corpus Statistics")

    if not dbs_exist:
        st.error("Please run Ingestion first.")
        return

    # Show database status summary
    render_db_stats_summary(config, db_t1, db_t2, period_t1_label, period_t2_label)

    st.write("Compare word frequencies between the two time periods.")
    report_top_n = st.number_input("Top N Shared Words", value=30, min_value=10)

    project_id = config.get("project_id", "")
    available_models = get_available_models(project_id) if project_id else []
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
            include_semantic_change,
            project_id
        )

    display_existing_report()


def generate_comparison_report_ui(
    db_t1: str, db_t2: str,
    period_t1_label: str, period_t2_label: str,
    top_n: int, model_name: str | None,
    include_semantic_change: bool,
    project_id: str
) -> None:
    """Generates and displays the comparison report."""

    try:
        from semantic_change.reporting import generate_comparison_report
        from semantic_change.corpus import CorpusManager

        manager = CorpusManager()
        manager.add_corpus(period_t1_label, os.path.dirname(db_t1), db_t1)
        manager.add_corpus(period_t2_label, os.path.dirname(db_t2), db_t2)

        report_path = os.path.join(OUTPUT_DIR, "processing_report.md")

        markdown_report, df = generate_comparison_report(
            manager.get_corpus(period_t1_label),
            manager.get_corpus(period_t2_label),
            top_n=top_n,
            output_path=report_path,
            model_name=model_name,
            include_semantic_change=include_semantic_change,
            return_dataframe=True,
            project_id=project_id,
            tqdm_class=stqdm,
        )

        # Store dataframe in session state for display
        st.session_state.report_dataframe = df
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

        # Read markdown for header info
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into header (before table) and table
        lines = content.split("\n")
        header_lines = []
        for line in lines:
            if line.startswith("| Lemma"):
                break
            header_lines.append(line)

        # Display header as markdown
        st.markdown("\n".join(header_lines))

        # Display table as sortable dataframe if available in session state
        if "report_dataframe" in st.session_state and st.session_state.report_dataframe is not None:
            st.dataframe(
                st.session_state.report_dataframe,
                width="stretch",
                hide_index=True,
            )
        else:
            # Fallback: parse markdown table to dataframe
            import pandas as pd
            table_lines = [l for l in lines if l.startswith("|") and not l.startswith("|---")]
            if len(table_lines) > 1:
                # Parse header
                headers = [h.strip() for h in table_lines[0].split("|")[1:-1]]
                # Parse rows
                rows = []
                for row_line in table_lines[1:]:
                    cells = [c.strip() for c in row_line.split("|")[1:-1]]
                    rows.append(cells)
                df = pd.DataFrame(rows, columns=headers)
                # Convert numeric columns
                for col in df.columns:
                    if col not in ["Lemma", "POS"]:
                        df[col] = pd.to_numeric(df[col].str.replace("%", ""), errors="coerce")
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                # No table found, show raw markdown
                st.markdown(content)


# =============================================================================
# Semantic Change Analysis Page
# =============================================================================

def render_search_parameters(config: dict) -> dict:
    """Renders the Search section parameters."""
    params = {}
    st.markdown("#### Search")

    # Target word input
    params["target_word"] = st.text_input("Target Word", value=config["target_word"])
    params["exact_match"] = st.checkbox(
        "Exact wordform",
        value=config.get("exact_match", False),
        help="If checked, search by exact token form (e.g., 'Glück' only matches 'Glück'). "
             "If unchecked, search by lemma (e.g., 'Glück' also matches 'Glücke', 'Glückes')."
    )

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
        help="Maximum embeddings to load from ChromaDB per time period."
    )

    params["viz_max_instances"] = st.number_input(
        "Max Points per Cluster (Viz)",
        min_value=10,
        max_value=2000,
        value=config.get("viz_max_instances", 100),
        help="Maximum points to display per sense cluster in visualization. "
             "Increase this to show more data points on the plot."
    )

    return params


def render_embedding_parameters(config: dict, available_models: list[str]) -> dict:
    """Renders the Embedding section parameters."""
    params = {}
    st.markdown("#### Embedding")

    # Model selection
    selected_safe_model = st.selectbox(
        "Select Embedding Set",
        available_models,
        index=0 if available_models else None,
        help="Select the pre-computed embeddings to use.",
    )
    params["selected_safe_model"] = selected_safe_model

    st.info(f"Using embeddings from: {selected_safe_model}")

    # Derive a likely model name from the safe model name for the tokenizer
    default_model = selected_safe_model if selected_safe_model else config["model_name"]
    params["model_name"] = st.text_input(
        "Model ID (for Tokenizer)",
        value=default_model,
        help="The HuggingFace model ID used for tokenization. Usually matches the embedding set.",
    )

    return params


def render_wsi_parameters(config: dict) -> dict:
    """Renders the Word Sense Induction section parameters."""
    params = {}
    st.markdown("#### Word Sense Induction")

    # WSI enable checkbox (default: yes)
    params["wsi_enabled"] = st.checkbox(
        "Enable Word Sense Induction",
        value=config.get("wsi_enabled", True),
        help="If enabled, cluster word usages into sense groups. If disabled, all usages are treated as one sense."
    )

    if params["wsi_enabled"]:
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

            # UMAP-specific parameters
            if params["clustering_reduction"] == "umap":
                st.markdown(
                    "_[UMAP parameters documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html)_"
                )
                params["umap_n_neighbors"] = st.slider(
                    "UMAP n_neighbors",
                    min_value=2,
                    max_value=200,
                    value=config.get("umap_n_neighbors", 15),
                    help="Controls local vs global structure. Low values (2-5) focus on fine detail, "
                         "high values (100+) emphasize global patterns. Default: 15."
                )
                params["umap_min_dist"] = st.slider(
                    "UMAP min_dist",
                    min_value=0.0,
                    max_value=0.99,
                    value=config.get("umap_min_dist", 0.1),
                    step=0.05,
                    help="How tightly points can be packed. Low values (0.0) create clumpy embeddings, "
                         "high values (0.8+) spread points apart. Default: 0.1."
                )
                umap_metric_options = ["euclidean", "manhattan", "cosine", "correlation"]
                current_metric = config.get("umap_metric", "euclidean")
                if current_metric not in umap_metric_options:
                    current_metric = "euclidean"
                params["umap_metric"] = st.selectbox(
                    "UMAP metric",
                    options=umap_metric_options,
                    index=umap_metric_options.index(current_metric),
                    help="Distance metric for computing nearness. 'cosine' is often good for embeddings."
                )
            # t-SNE specific parameters
            elif params["clustering_reduction"] == "tsne":
                st.markdown(
                    "_[How to use t-SNE effectively](https://distill.pub/2016/misread-tsne/)_"
                )
                params["tsne_perplexity"] = st.slider(
                    "t-SNE perplexity",
                    min_value=2,
                    max_value=100,
                    value=config.get("tsne_perplexity", 30),
                    help="Related to the number of nearest neighbors. Larger datasets usually require "
                         "larger perplexity (5-50 typical). Different values can result in significantly "
                         "different results. Must be less than the number of samples."
                )
            else:
                params["tsne_perplexity"] = config.get("tsne_perplexity", 30)

            # Set defaults for non-selected methods
            if params["clustering_reduction"] != "umap":
                params["umap_n_neighbors"] = config.get("umap_n_neighbors", 15)
                params["umap_min_dist"] = config.get("umap_min_dist", 0.1)
                params["umap_metric"] = config.get("umap_metric", "euclidean")
            if params["clustering_reduction"] != "tsne":
                params["tsne_perplexity"] = config.get("tsne_perplexity", 30)
        else:
            params["clustering_n_components"] = config.get("clustering_n_components", 50)
            params["umap_n_neighbors"] = config.get("umap_n_neighbors", 15)
            params["umap_min_dist"] = config.get("umap_min_dist", 0.1)
            params["umap_metric"] = config.get("umap_metric", "euclidean")
            params["tsne_perplexity"] = config.get("tsne_perplexity", 30)

        # Clustering algorithm
        wsi_options = ["hdbscan", "kmeans", "spectral", "agglomerative", "substitute"]
        current_algo = config["wsi_algorithm"]
        if current_algo not in wsi_options:
            current_algo = "hdbscan"
        params["wsi_algorithm"] = st.selectbox(
            "Clustering Algorithm",
            options=wsi_options,
            index=wsi_options.index(current_algo),
        )

        # Show info message and filter model selector for substitute-based WSI
        if params["wsi_algorithm"] == "substitute":
            st.info(
                "ℹ️ **Substitute-based WSI** (Eyal et al., 2022): Uses MLM top-k substitutes "
                "and graph community detection instead of embedding clustering. "
                "Senses are represented by interpretable word lists. "
                "Number of senses is determined automatically by Louvain algorithm."
            )

            # Filter model selector for substitute WSI
            filter_model_options = [
                "en_core_web_sm",
                "en_core_web_lg",
                "de_core_news_sm",
                "de_core_news_lg",
                "Other (Custom)",
            ]
            current_filter = config.get("substitute_filter_model", "en_core_web_sm")
            if current_filter and current_filter not in filter_model_options[:-1]:
                default_idx = filter_model_options.index("Other (Custom)")
            elif current_filter:
                default_idx = filter_model_options.index(current_filter)
            else:
                default_idx = 0

            selected_filter = st.selectbox(
                "Filter Model (for stopwords/lemmatization)",
                filter_model_options,
                index=default_idx,
                help="spaCy model used for filtering stopwords and lemmatizing substitutes. "
                     "Use a model appropriate for your corpus language/time period."
            )

            if selected_filter == "Other (Custom)":
                params["substitute_filter_model"] = st.text_input(
                    "Custom spaCy Model",
                    value=current_filter if current_filter not in filter_model_options[:-1] else "",
                    help="e.g., fr_core_news_sm, es_core_news_lg"
                )
            else:
                params["substitute_filter_model"] = selected_filter

            # Merge threshold for combining similar communities
            params["substitute_merge_threshold"] = st.slider(
                "Merge Threshold (Jaccard)",
                min_value=0.0,
                max_value=0.6,
                value=config.get("substitute_merge_threshold", 0.3),
                step=0.05,
                help="Communities with Jaccard similarity >= threshold on their top representatives "
                     "will be merged. 0.0 = no merging. Typical values: 0.2 (conservative) to 0.4 (aggressive)."
            )
        else:
            params["substitute_filter_model"] = config.get("substitute_filter_model")
            params["substitute_merge_threshold"] = config.get("substitute_merge_threshold", 0.0)

        if params["wsi_algorithm"] == "hdbscan":
            params["min_cluster_size"] = st.number_input(
                "Min Cluster Size",
                min_value=2,
                value=config["min_cluster_size"],
            )
            params["n_clusters"] = config["n_clusters"]
        elif params["wsi_algorithm"] == "substitute":
            # Substitute WSI determines clusters automatically via Louvain
            # min_cluster_size is used for min_community_size
            params["min_cluster_size"] = st.number_input(
                "Min Community Size",
                min_value=2,
                value=config["min_cluster_size"],
                help="Minimum number of words to form a valid sense community."
            )
            params["n_clusters"] = config["n_clusters"]
        else:
            # kmeans, spectral, agglomerative - need explicit n_clusters
            params["n_clusters"] = st.number_input(
                "Number of Clusters (k)",
                min_value=2,
                value=config["n_clusters"],
            )
            params["min_cluster_size"] = config["min_cluster_size"]
    else:
        # Set default values when WSI is disabled
        params["clustering_reduction"] = config.get("clustering_reduction", "None")
        params["clustering_n_components"] = config.get("clustering_n_components", 50)
        params["umap_n_neighbors"] = config.get("umap_n_neighbors", 15)
        params["umap_min_dist"] = config.get("umap_min_dist", 0.1)
        params["umap_metric"] = config.get("umap_metric", "euclidean")
        params["tsne_perplexity"] = config.get("tsne_perplexity", 30)
        params["wsi_algorithm"] = config["wsi_algorithm"]
        params["n_clusters"] = config["n_clusters"]
        params["min_cluster_size"] = config["min_cluster_size"]

    return params


def render_visualization_parameters(config: dict) -> dict:
    """Renders the Visualization section parameters."""
    params = {}
    st.markdown("#### Visualization")

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

    # Semantic neighbors
    params["k_neighbors"] = st.number_input(
        "Semantic Neighbors (k)",
        min_value=1,
        value=config["k_neighbors"],
    )

    return params


def update_config_from_params(config: dict, params: dict) -> None:
    """Updates the config dict with values from params."""
    config["target_word"] = params.get("target_word", config["target_word"])
    config["pos_filter"] = params.get("pos_filter", config["pos_filter"])
    config["n_samples"] = params.get("n_samples", config["n_samples"])
    config["viz_max_instances"] = params.get("viz_max_instances", config.get("viz_max_instances", 100))
    config["model_name"] = params.get("model_name", config["model_name"])
    config["wsi_enabled"] = params.get("wsi_enabled", config.get("wsi_enabled", True))
    config["wsi_algorithm"] = params.get("wsi_algorithm", config["wsi_algorithm"])
    config["min_cluster_size"] = params.get("min_cluster_size", config["min_cluster_size"])
    config["n_clusters"] = params.get("n_clusters", config["n_clusters"])
    config["clustering_reduction"] = params.get("clustering_reduction", config["clustering_reduction"])
    config["clustering_n_components"] = params.get("clustering_n_components", config["clustering_n_components"])
    config["umap_n_neighbors"] = params.get("umap_n_neighbors", config.get("umap_n_neighbors", 15))
    config["umap_min_dist"] = params.get("umap_min_dist", config.get("umap_min_dist", 0.1))
    config["umap_metric"] = params.get("umap_metric", config.get("umap_metric", "euclidean"))
    config["tsne_perplexity"] = params.get("tsne_perplexity", config.get("tsne_perplexity", 30))
    config["viz_reduction"] = params.get("viz_reduction", config["viz_reduction"])
    config["k_neighbors"] = params.get("k_neighbors", config["k_neighbors"])
    config["substitute_filter_model"] = params.get("substitute_filter_model", config.get("substitute_filter_model"))
    config["substitute_merge_threshold"] = params.get("substitute_merge_threshold", config.get("substitute_merge_threshold", 0.0))


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

            with capture_output(update_logs):
                results = run_single_analysis(
                    project_id=config["project_id"],
                    target_word=params["target_word"],
                    db_path_t1=db_t1,
                    db_path_t2=db_t2,
                    period_t1_label=period_t1_label,
                    period_t2_label=period_t2_label,
                    model_name=params["model_name"],
                    k_neighbors=params["k_neighbors"],
                    min_cluster_size=params["min_cluster_size"],
                    n_clusters=params["n_clusters"],
                    wsi_enabled=params.get("wsi_enabled", True),
                    wsi_algorithm=params["wsi_algorithm"],
                    pos_filter=pos_filter,
                    clustering_reduction=clust_red,
                    clustering_n_components=params["clustering_n_components"],
                    umap_n_neighbors=params.get("umap_n_neighbors", 15),
                    umap_min_dist=params.get("umap_min_dist", 0.1),
                    umap_metric=params.get("umap_metric", "euclidean"),
                    tsne_perplexity=params.get("tsne_perplexity", 30),
                    viz_reduction=params["viz_reduction"],
                    n_samples=params["n_samples"],
                    viz_max_instances=params["viz_max_instances"],
                    exact_match=params.get("exact_match", False),
                    substitute_filter_model=params.get("substitute_filter_model"),
                    substitute_merge_threshold=params.get("substitute_merge_threshold", 0.0),
                )

            st.success("Analysis Complete!")

            # Store analysis results and metadata in session state for drill-down
            if results is not None:
                st.session_state['analysis_results'] = results
                st.session_state['analysis_metadata'] = {
                    'project_id': config["project_id"],
                    'model_name': params["model_name"],
                    'target_word': params["target_word"],
                    'pos_filter': pos_filter,
                    'n_samples': params["n_samples"],
                    'wsi_algorithm': params["wsi_algorithm"],
                }

            # Store visualization parameters in session state for persistence
            st.session_state['last_viz_params'] = {
                'project_id': config["project_id"],
                'model_name': params["model_name"],
                'target_word': params["target_word"]
            }
            # Display visualizations immediately after analysis
            display_visualizations(
                project_id=config["project_id"],
                model_name=params["model_name"],
                target_word=params["target_word"]
            )

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)


def save_cluster_for_drilldown(cluster_id: int) -> str:
    """
    Save a sense cluster's data for drill-down analysis.

    Args:
        cluster_id: The cluster ID to save

    Returns:
        Path to the saved .npz file
    """
    results = st.session_state['analysis_results']
    metadata = st.session_state['analysis_metadata']

    # Use ClusterService to save the data
    return ClusterService.save_for_drilldown(
        embeddings=results['combined_embeddings'],
        sentences=results['combined_sentences'],
        filenames=results['combined_filenames'],
        time_labels=results['combined_time_labels'],
        sense_labels=results['sense_labels'],
        spans=results['combined_spans'],
        metadata=metadata,
        cluster_id=cluster_id,
        output_dir=OUTPUT_DIR
    )


def display_visualizations(
    project_id: str = None,
    model_name: str = None,
    target_word: str = None
) -> None:
    """
    Displays the generated visualization HTML files.

    Args:
        project_id: 4-digit project identifier (for finding files with new naming)
        model_name: HuggingFace model name (for finding files with new naming)
        target_word: The word being analyzed (for finding files with new naming)
    """
    st.subheader("Visualizations")

    # Determine filename prefix
    if project_id and model_name and target_word:
        from main import get_model_short_name
        model_short = get_model_short_name(model_name)
        prefix = f"k{project_id}_{model_short}_{target_word}_"
    else:
        prefix = ""

    viz_files = [
        ("time_period.html", "⏳ Time Period Clustering"),
        ("sense_clusters.html", "🧩 Sense Clusters"),
        ("sense_time_combined.html", "🎨 Sense × Time (Combined)"),
        ("substitute_graph.html", "🕸️ Substitute Co-occurrence Graph"),
    ]

    for base_filename, title in viz_files:
        filename = f"{prefix}{base_filename}" if prefix else base_filename
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            st.markdown(f"### {title}")
            with open(path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)

            # Add drill-down UI after sense_clusters.html
            if base_filename == "sense_clusters.html" and 'analysis_results' in st.session_state:
                results = st.session_state['analysis_results']
                unique_clusters = sorted(set(results['sense_labels']))

                st.markdown("#### Save Cluster for Drill Down")
                col_select, col_button = st.columns([2, 1])
                with col_select:
                    selected_cluster = st.selectbox(
                        "Select Cluster",
                        options=unique_clusters,
                        format_func=lambda x: f"Cluster {x} ({sum(results['sense_labels'] == x)} instances)",
                        key="drilldown_cluster_select"
                    )
                with col_button:
                    if st.button("Save for Drill Down", key="save_drilldown_btn"):
                        filepath = save_cluster_for_drilldown(selected_cluster)
                        st.success(f"Saved to `{filepath}`")
                        st.info("Go to **SCA Micro** page to analyze this cluster.")

    # Neighbor Graph files (shown first)
    if prefix:
        neighbor_graph_pattern = os.path.join(OUTPUT_DIR, f"{prefix}neighbors_graph_cluster_*.html")
    else:
        neighbor_graph_pattern = os.path.join(OUTPUT_DIR, "neighbors_graph_cluster_*.html")

    neighbor_graph_files = sorted(glob.glob(neighbor_graph_pattern))
    if neighbor_graph_files:
        st.markdown("### 🕸️ Semantic Neighbors Graph")
        for nf in neighbor_graph_files:
            # Extract cluster name from filename
            basename = os.path.basename(nf)
            # Remove prefix if present
            if prefix:
                basename = basename.replace(prefix, "")
            cluster_name = (
                basename
                .replace("neighbors_graph_", "")
                .replace(".html", "")
                .replace("_", " ")
                .title()
            )
            st.markdown(f"**{cluster_name}**")
            with open(nf, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)

    # Neighbor files (shown second)
    if prefix:
        neighbor_pattern = os.path.join(OUTPUT_DIR, f"{prefix}neighbors_cluster_*.html")
    else:
        neighbor_pattern = os.path.join(OUTPUT_DIR, "neighbors_cluster_*.html")

    neighbor_files = sorted(glob.glob(neighbor_pattern))
    if neighbor_files:
        st.markdown("### 🕸️ Semantic Neighbors")
        for nf in neighbor_files:
            # Extract cluster name from filename
            basename = os.path.basename(nf)
            # Remove prefix if present
            if prefix:
                basename = basename.replace(prefix, "")
            cluster_name = (
                basename
                .replace("neighbors_", "")
                .replace(".html", "")
                .replace("_", " ")
                .title()
            )
            st.markdown(f"**{cluster_name}**")
            with open(nf, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)

    # Archive button
    if prefix:
        st.markdown("---")
        if st.button("📦 Archive Results", help="Save all visualization files to a zip archive"):
            archive_visualizations(prefix, project_id, model_name, target_word)


def archive_visualizations(
    prefix: str,
    project_id: str,
    model_name: str,
    target_word: str
) -> None:
    """
    Archive all visualization HTML files matching the prefix into a zip file.

    Args:
        prefix: The filename prefix to match
        project_id: Project ID for the archive name
        model_name: Model name for the archive name
        target_word: Target word for the archive name
    """
    # Create archive directory if it doesn't exist
    archive_dir = os.path.join(OUTPUT_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    # Collect all HTML files matching the prefix
    html_files = glob.glob(os.path.join(OUTPUT_DIR, f"{prefix}*.html"))

    if not html_files:
        st.warning("No visualization files found to archive.")
        return

    # Generate archive filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    from main import get_model_short_name
    model_short = get_model_short_name(model_name)
    archive_name = f"k{project_id}_{model_short}_{target_word}_{timestamp}.zip"
    archive_path = os.path.join(archive_dir, archive_name)

    # Create zip file
    try:
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for html_file in html_files:
                # Add file with just the basename (no directory structure)
                arcname = os.path.basename(html_file)
                zipf.write(html_file, arcname)

        st.success(f"Archived {len(html_files)} files to `{archive_path}`")
    except Exception as e:
        st.error(f"Failed to create archive: {e}")


def render_dashboard_page(
    config: dict,
    db_t1: str, db_t2: str,
    period_t1_label: str, period_t2_label: str,
    dbs_exist: bool
) -> None:
    """Renders the Semantic Change Analysis page."""
    st.title("🔎 Semantic Change Analysis")

    if not dbs_exist:
        st.error("Databases missing. Please go to 'Data Ingestion' first.")
        return

    project_id = config.get("project_id", "")
    available_models = get_available_models(project_id) if project_id else []
    if not available_models:
        st.warning("⚠️ No embeddings found. Please go to the **Embeddings & Models** tab to generate them first.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")
        params = render_search_parameters(config)
        params.update(render_embedding_parameters(config, available_models))
        params.update(render_wsi_parameters(config))
        params.update(render_visualization_parameters(config))

        update_config_from_params(config, params)
        run_btn = st.button("Run Analysis", type="primary")

    with col2:
        if run_btn:
            save_config(config)
            run_analysis(config, params, db_t1, db_t2, period_t1_label, period_t2_label)

        # Display visualizations if we have stored parameters (persists across reruns)
        elif 'last_viz_params' in st.session_state:
            viz_params = st.session_state['last_viz_params']
            display_visualizations(
                project_id=viz_params['project_id'],
                model_name=viz_params['model_name'],
                target_word=viz_params['target_word']
            )


# =============================================================================
# SCA Micro (Drill-Down) Page
# =============================================================================

def load_cluster_data(filepath: str) -> None:
    """Load cluster data from .npz file into session state."""
    import numpy as np

    try:
        data = np.load(filepath, allow_pickle=True)
        metadata = json.loads(str(data['metadata']))

        st.session_state['micro_cluster_data'] = {
            'embeddings': data['embeddings'],
            'sentences': data['sentences'],
            'filenames': data['filenames'],
            'time_labels': data['time_labels'],
            'spans': data['spans'],
            'metadata': metadata,
            'filepath': filepath,
        }
    except Exception as e:
        st.error(f"Failed to load cluster file: {e}")


def display_micro_cluster_info() -> None:
    """Display loaded cluster metadata."""
    if 'micro_cluster_data' not in st.session_state:
        return

    data = st.session_state['micro_cluster_data']
    metadata = data['metadata']

    st.markdown("#### Loaded Cluster Info")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Instances", len(data['embeddings']))
    with col2:
        st.metric("Original Cluster", metadata.get('original_cluster_id', '?'))
    with col3:
        st.metric("Target Word", metadata.get('target_word', '?'))

    with st.expander("Full Metadata"):
        st.json(metadata)


def render_micro_wsi_parameters(config: dict) -> dict:
    """Renders WSI parameters for micro analysis."""
    params = {}
    st.markdown("#### Micro WSI Parameters")

    # Pre-clustering reduction
    clust_options = ["None", "pca", "umap", "tsne"]
    current_clust = config.get("clustering_reduction", "None")
    if current_clust not in clust_options:
        current_clust = "None"

    params["clustering_reduction"] = st.selectbox(
        "Pre-clustering Reduction",
        options=clust_options,
        index=clust_options.index(current_clust),
        help="Reduce dimensions before clustering.",
        key="micro_clust_reduction"
    )

    if params["clustering_reduction"] != "None":
        params["clustering_n_components"] = st.number_input(
            "Clustering Reduction Dims",
            min_value=2,
            max_value=200,
            value=config.get("clustering_n_components", 50),
            key="micro_clust_dims"
        )

        if params["clustering_reduction"] == "umap":
            params["umap_n_neighbors"] = st.slider(
                "UMAP n_neighbors",
                min_value=2,
                max_value=200,
                value=config.get("umap_n_neighbors", 15),
                key="micro_umap_neighbors"
            )
            params["umap_min_dist"] = st.slider(
                "UMAP min_dist",
                min_value=0.0,
                max_value=0.99,
                value=config.get("umap_min_dist", 0.1),
                step=0.05,
                key="micro_umap_dist"
            )
            umap_metric_options = ["euclidean", "manhattan", "cosine", "correlation"]
            current_metric = config.get("umap_metric", "euclidean")
            if current_metric not in umap_metric_options:
                current_metric = "euclidean"
            params["umap_metric"] = st.selectbox(
                "UMAP metric",
                options=umap_metric_options,
                index=umap_metric_options.index(current_metric),
                key="micro_umap_metric"
            )
        elif params["clustering_reduction"] == "tsne":
            params["tsne_perplexity"] = st.slider(
                "t-SNE perplexity",
                min_value=2,
                max_value=100,
                value=config.get("tsne_perplexity", 30),
                key="micro_tsne_perp"
            )
    else:
        params["clustering_n_components"] = config.get("clustering_n_components", 50)
        params["umap_n_neighbors"] = config.get("umap_n_neighbors", 15)
        params["umap_min_dist"] = config.get("umap_min_dist", 0.1)
        params["umap_metric"] = config.get("umap_metric", "euclidean")
        params["tsne_perplexity"] = config.get("tsne_perplexity", 30)

    # Clustering algorithm
    wsi_options = ["hdbscan", "kmeans", "spectral", "agglomerative", "substitute"]
    current_algo = config["wsi_algorithm"]
    if current_algo not in wsi_options:
        current_algo = "hdbscan"
    params["wsi_algorithm"] = st.selectbox(
        "Clustering Algorithm",
        options=wsi_options,
        index=wsi_options.index(current_algo),
        key="micro_wsi_algo"
    )

    if params["wsi_algorithm"] == "substitute":
        st.info("Substitute-based WSI uses MLM predictions for clustering.")
        filter_model_options = [
            "en_core_web_sm", "en_core_web_lg",
            "de_core_news_sm", "de_core_news_lg",
            "Other (Custom)",
        ]
        current_filter = config.get("substitute_filter_model", "en_core_web_sm")
        if current_filter and current_filter not in filter_model_options[:-1]:
            default_idx = filter_model_options.index("Other (Custom)")
        elif current_filter:
            default_idx = filter_model_options.index(current_filter)
        else:
            default_idx = 0

        selected_filter = st.selectbox(
            "Filter Model",
            filter_model_options,
            index=default_idx,
            key="micro_filter_model"
        )
        if selected_filter == "Other (Custom)":
            params["substitute_filter_model"] = st.text_input(
                "Custom spaCy Model",
                value=current_filter if current_filter not in filter_model_options[:-1] else "",
                key="micro_custom_filter"
            )
        else:
            params["substitute_filter_model"] = selected_filter

        params["substitute_merge_threshold"] = st.slider(
            "Merge Threshold",
            min_value=0.0,
            max_value=0.6,
            value=config.get("substitute_merge_threshold", 0.3),
            step=0.05,
            key="micro_merge_thresh"
        )
    else:
        params["substitute_filter_model"] = config.get("substitute_filter_model")
        params["substitute_merge_threshold"] = config.get("substitute_merge_threshold", 0.0)

    if params["wsi_algorithm"] == "hdbscan":
        params["min_cluster_size"] = st.number_input(
            "Min Cluster Size",
            min_value=2,
            value=config["min_cluster_size"],
            key="micro_min_cluster"
        )
        params["n_clusters"] = config["n_clusters"]
    elif params["wsi_algorithm"] == "substitute":
        params["min_cluster_size"] = st.number_input(
            "Min Community Size",
            min_value=2,
            value=config["min_cluster_size"],
            key="micro_min_community"
        )
        params["n_clusters"] = config["n_clusters"]
    else:
        params["n_clusters"] = st.number_input(
            "Number of Clusters (k)",
            min_value=2,
            value=config["n_clusters"],
            key="micro_n_clusters"
        )
        params["min_cluster_size"] = config["min_cluster_size"]

    # Visualization
    st.markdown("#### Visualization")
    viz_options = ["pca", "umap", "tsne"]
    current_viz = config.get("viz_reduction", "pca")
    if current_viz not in viz_options:
        current_viz = "pca"
    params["viz_reduction"] = st.selectbox(
        "Visualization Reduction",
        options=viz_options,
        index=viz_options.index(current_viz),
        key="micro_viz_reduction"
    )

    params["viz_max_instances"] = st.number_input(
        "Max Points per Cluster",
        min_value=10,
        max_value=2000,
        value=config.get("viz_max_instances", 100),
        key="micro_viz_max"
    )

    return params


def run_micro_analysis_ui(params: dict) -> None:
    """Run micro analysis and display results."""
    if 'micro_cluster_data' not in st.session_state:
        st.error("No cluster data loaded.")
        return

    data = st.session_state['micro_cluster_data']
    filepath = data['filepath']

    # Debug: show parameters being used
    st.write("**Parameters being used:**")
    st.write(f"- clustering_reduction: `{params.get('clustering_reduction')}`")
    st.write(f"- wsi_algorithm: `{params.get('wsi_algorithm')}`")
    st.write(f"- viz_reduction: `{params.get('viz_reduction')}`")

    logs = []
    log_area = st.empty()

    def update_logs(text):
        logs.append(text)
        log_area.code("".join(logs))

    with st.spinner("Running micro analysis..."):
        try:
            from main import run_micro_analysis

            clust_red = params["clustering_reduction"] if params["clustering_reduction"] != "None" else None

            with capture_output(update_logs):
                results = run_micro_analysis(
                    cluster_file=filepath,
                    wsi_algorithm=params["wsi_algorithm"],
                    min_cluster_size=params["min_cluster_size"],
                    n_clusters=params["n_clusters"],
                    clustering_reduction=clust_red,
                    clustering_n_components=params.get("clustering_n_components", 50),
                    umap_n_neighbors=params.get("umap_n_neighbors", 15),
                    umap_min_dist=params.get("umap_min_dist", 0.1),
                    umap_metric=params.get("umap_metric", "euclidean"),
                    tsne_perplexity=params.get("tsne_perplexity", 30),
                    viz_reduction=params["viz_reduction"],
                    viz_max_instances=params["viz_max_instances"],
                    output_dir=OUTPUT_DIR,
                    substitute_filter_model=params.get("substitute_filter_model"),
                    substitute_merge_threshold=params.get("substitute_merge_threshold", 0.0),
                )

            if results:
                st.session_state['micro_viz_prefix'] = results['viz_prefix']
                st.session_state['micro_analysis_success'] = True
                st.rerun()  # Refresh to show new visualizations
            else:
                st.error("Micro analysis failed.")

        except Exception as e:
            st.error(f"Micro analysis failed: {e}")
            st.exception(e)


def display_micro_visualizations() -> None:
    """Display micro analysis visualization files."""
    if 'micro_viz_prefix' not in st.session_state:
        return

    prefix = st.session_state['micro_viz_prefix']

    st.subheader("Micro Analysis Visualizations")

    # Debug: show prefix and file modification times
    st.caption(f"Prefix: `{prefix}`")

    viz_files = [
        ("time_period.html", "Time Period (Micro)"),
        ("sense_clusters.html", "Sub-Sense Clusters (Micro)"),
        ("sense_time_combined.html", "Sub-Sense × Time (Micro)"),
        ("substitute_graph.html", "Substitute Graph (Micro)"),
    ]

    for base_filename, title in viz_files:
        filename = f"{prefix}{base_filename}"
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            # Debug: show file modification time
            import datetime
            mtime = os.path.getmtime(path)
            mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
            st.markdown(f"### {title}")
            st.caption(f"File modified: {mtime_str}")
            with open(path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)

    # Display neighbor visualizations
    neighbor_pattern = os.path.join(OUTPUT_DIR, f"{prefix}neighbors_cluster_*.html")
    neighbor_files = sorted(glob.glob(neighbor_pattern))
    if neighbor_files:
        st.markdown("### 📊 Sub-Cluster Neighbors")
        for nf in neighbor_files:
            basename = os.path.basename(nf)
            cluster_name = basename.replace(prefix, "").replace("neighbors_", "").replace(".html", "").replace("_", " ").title()
            st.markdown(f"**{cluster_name}**")
            with open(nf, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=400, scrolling=True)

    # Display neighbor graph visualizations
    graph_pattern = os.path.join(OUTPUT_DIR, f"{prefix}neighbors_graph_cluster_*.html")
    graph_files = sorted(glob.glob(graph_pattern))
    if graph_files:
        st.markdown("### 🕸️ Sub-Cluster Neighbor Graphs")
        for gf in graph_files:
            basename = os.path.basename(gf)
            cluster_name = basename.replace(prefix, "").replace("neighbors_graph_", "").replace(".html", "").replace("_", " ").title()
            st.markdown(f"**{cluster_name}**")
            with open(gf, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)


def get_cluster_file_label(filepath: str) -> str:
    """
    Parse cluster file metadata and return a user-friendly label.
    Format: "word (project_id, model, cluster N)"
    """
    import numpy as np

    try:
        data = np.load(filepath, allow_pickle=True)
        metadata = json.loads(str(data['metadata']))

        word = metadata.get('target_word', '?')
        project_id = metadata.get('project_id', '?')
        model = metadata.get('model_name', '?')
        cluster_id = metadata.get('original_cluster_id', '?')
        n_instances = len(data['embeddings'])

        # Shorten model name for display
        if '/' in model:
            model = model.split('/')[-1]
        if len(model) > 20:
            model = model[:17] + "..."

        return f"{word} (k{project_id}, {model}, cluster {cluster_id}, n={n_instances})"
    except Exception:
        return os.path.basename(filepath)


def render_micro_page(config: dict) -> None:
    """Renders the SCA Micro (Drill-Down) page."""
    st.title("🔬 SCA Micro - Drill Down Analysis")
    st.markdown("""
    Analyze a specific sense cluster in more detail. First save a cluster from
    SCA Macro, then load it here for deeper analysis.
    """)

    # Check for saved clusters
    clusters_dir = os.path.join(OUTPUT_DIR, "clusters")

    if not os.path.exists(clusters_dir):
        st.warning("No saved clusters found. Run SCA Macro first and save a cluster for drill-down.")
        return

    cluster_files = sorted(glob.glob(os.path.join(clusters_dir, "*.npz")))

    if not cluster_files:
        st.warning("No saved clusters found. Run SCA Macro first and save a cluster for drill-down.")
        return

    # Pre-load labels for all cluster files (cached in session state)
    if 'cluster_file_labels' not in st.session_state:
        st.session_state['cluster_file_labels'] = {}

    for f in cluster_files:
        if f not in st.session_state['cluster_file_labels']:
            st.session_state['cluster_file_labels'][f] = get_cluster_file_label(f)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Load Cluster")

        # File selector with parsed metadata labels
        selected_file = st.selectbox(
            "Select Cluster File",
            options=cluster_files,
            format_func=lambda x: st.session_state['cluster_file_labels'].get(x, os.path.basename(x)),
            key="micro_file_select"
        )

        if st.button("Load Cluster", key="load_cluster_btn"):
            load_cluster_data(selected_file)
            st.rerun()

        # Display loaded cluster info
        if 'micro_cluster_data' in st.session_state:
            display_micro_cluster_info()

            st.markdown("---")
            micro_params = render_micro_wsi_parameters(config)

            if st.button("Run Micro Analysis", type="primary", key="run_micro_btn"):
                run_micro_analysis_ui(micro_params)

    with col2:
        # Show success message after rerun
        if st.session_state.get('micro_analysis_success'):
            st.success("Micro Analysis Complete!")
            del st.session_state['micro_analysis_success']

        # Display visualizations if available
        display_micro_visualizations()


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

    # Ensure project_id exists
    project_id = ensure_project_id(config)

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
    elif page == "SCA Macro":
        render_dashboard_page(config, db_t1, db_t2, period_t1_label, period_t2_label, dbs_exist)
    elif page == "SCA Micro":
        render_micro_page(config)


# Run the app
main()
