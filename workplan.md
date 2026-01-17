# Work Plan

## I. Major Feature: HPC & GPU Cluster Support (SLURM/Apptainer)
Enable offloading of heavy computational tasks (Ingestion and Embedding Generation) to a remote SLURM-based GPU cluster using Apptainer containers.

Documentation about the cluster can be found here: https://doku.hpc.uni-wuerzburg.de (Pay especially attention where to put the container and pay attention to the restrictions on the login node)
How to use a slurm container here:
https://doku.hpc.uni-wuerzburg.de/slurm-container/

### 1. Containerization (`hpc/`)
- [x] **Apptainer Definition File**: Create `hpc/sild.def` to build a portable environment.
    - Base Image: Official Python or NVIDIA PyTorch image.
    - Setup: Install system dependencies, `uv`, project dependencies (from `requirements.txt`), and download necessary spaCy models (`en_core_web_sm`, `en_core_web_lg` and also the German models!).
    - **Note**: Ensure the container uses `uv` for fast, reproducible installs.

### 2. Remote Management Logic (`src/hpc/`)
- [x] **Cluster Configuration**: Create a config handler (e.g., `hpc_config.json`) to store:
    - Hostname, Username (SSH key auth assumed).
    - Remote paths (work directory, data directory).
    - SLURM defaults (partition, time limit, GPU count).
- [x] **Data Synchronization**: Implement `rsync` wrappers to:
    - Push: Codebase (`src/`), Raw Data (for ingestion), or SQLite DBs (for embedding).
    - Pull: Generated SQLite DBs or ChromaDB folders.
- [x] **Job Submission Generator**: Create a utility to generate `.slurm` scripts dynamically.
    - Directives: `#SBATCH --partition=...`, `--gres=gpu:1`, `--cpus-per-task=...`.
    - Command: `apptainer exec --nv sild.sif uv run python ...`

### 3. CLI Integration
- [x] **New CLI Commands**: Add commands to a main entry point (e.g., `sild-hpc`) or subcommands:
    - `sild hpc push`: Sync code and data.
    - `sild hpc build`: Trigger container build (remotely or locally).
    - `sild hpc submit-ingest`: Submit ingestion job.
    - `sild hpc submit-embed`: Submit embedding job.
    - `sild hpc pull`: Retrieve results.

### 4. Documentation
- [x] **HPC Guide**: Add `docs/hpc_guide.md` explaining how to configure the SSH connection, build the container, and manage jobs.

### 5. Improvements
- [ ] **Externalize spaCy models for HPC**:
    - Instead of baking models into `sild.sif`, download them to a shared persistent directory (e.g., `data/models`) on the cluster.
    - Configure the container to look for models there (via `SPACY_DATA_PATH` or symlinks).
    - **Benefit**: Allows adding new languages without rebuilding the huge container image.

## II. Refactoring BertEmbedder & Embedding Performance Optimization ✅
Probably this can be done together with item one

**Status: Core optimizations complete (Jan 2026)**

### 1. Lazy Loading (Quick Win)
- [x] **Lazy Load BertEmbedder Components**:
    - Currently, `BertEmbedder` initializes the **MLM Head** (for neighbor discovery) and the **spaCy Filter Model** (for cleaning neighbors) in its `__init__` method.
    - This happens even for Batch Analysis (Embedding Generation), where these components are not needed, causing unnecessary memory usage and startup delay.
    - **Goal**: Refactor `BertEmbedder` to load `self.mlm_model` and `self.filter_nlp` only when `get_nearest_neighbors()` is called for the first time.
    - **Benefit**: Faster startup for batch processing and reduced memory footprint.
    - **Implemented**: Added `_ensure_mlm_model()` and `_ensure_filter_nlp()` methods with property accessors for backward compatibility.

### 2. Embedding Extraction Performance Optimizations

The following optimizations target `batch_extract()` in `embedding.py` and `process_corpus()` in `embeddings_generation.py`:

| Optimization | Impact | Effort | Status | Description |
|--------------|--------|--------|--------|-------------|
| Mixed precision (bf16/fp16) | **High** | Low | ✅ Done | Enable automatic mixed precision for ~2x throughput on modern GPUs (L40, A100, etc.) |
| Skip MLM/filter model | Medium | Low | ✅ Done | Implemented via lazy loading - models only loaded on first `get_nearest_neighbors()` call |
| Increase batch size | Medium | Low | ✅ Done | Auto-detects optimal batch size based on GPU VRAM (512 for 40GB+, 256 for 24GB+, etc.) |
| Keep tensors on GPU | Medium | Medium | ✅ Done | Use DLPack for zero-copy cupy→torch transfer in `_combine_layers()` |
| Direct HuggingFace usage | **High** | High | Pending | Replace spacy-transformers with direct transformers library, handle subword alignment manually |
| DataLoader prefetching | Medium | Medium | Pending | Pre-tokenize next batch while current batch is on GPU |

#### Implementation Summary (Completed):
1. ✅ **Mixed precision (bf16/fp16)** - `detect_optimal_dtype()` in `embedding.py` auto-selects bf16 for datacenter GPUs, fp16 for consumer GPUs
2. ✅ **Lazy loading** - `_ensure_mlm_model()` and `_ensure_filter_nlp()` methods load models only on first `get_nearest_neighbors()` call
3. ✅ **Auto batch size** - `detect_optimal_batch_size()` selects 512/256/128/64 based on GPU VRAM, CLI flag `--batch-size` available
4. ✅ **GPU tensor optimization** - `_combine_layers()` uses DLPack for zero-copy cupy→torch transfer

#### Remaining (Lower Priority):
5. **Direct HuggingFace** (long-term) - Major refactor but removes spacy-transformers overhead entirely.
6. **DataLoader prefetching** - Pre-tokenize next batch while current batch is on GPU.

### III. Refactoring: Separation of Concerns & Cleanup

- [ ] **Remove Custom Words Option**:
    - Remove the capability to compute embeddings for specific manual words, focusing instead on frequency-based batch processing.
    - **Update GUI**: Remove the "Custom Words" text area from `src/gui_app.py` (`render_create_embeddings_tab`).
    - **Update Logic**: Remove the `additional_words` parameter from `run_batch_generation` in `src/semantic_change/embeddings_generation.py`.
    - **Update CLI**: Remove the `--words` argument from the CLI parser.

- [ ] **Implement MVC Architecture (Clear Separation of Concerns)**:
    - **Goal**: Move all business logic out of `src/gui_app.py` into dedicated service/controller modules in `src/semantic_change/` or a new `src/controllers/` package. The GUI should only handle rendering and user input.
    - **Tasks**:
        - **Stats Service**: Move logic from `render_db_stats_summary` (direct DB/Chroma queries) to a new `src/semantic_change/services.py` or `reporting.py`.
        - **Dependency Management**: Move `check_spacy_transformer_deps` to a utility module (e.g., `src/utils/dependencies.py`).
        - **Embedding Management**: Move `delete_model_embeddings` logic to `src/semantic_change/vector_store.py` or a manager class.
        - **Configuration**: Ensure `load_config`/`save_config` are handled by a dedicated config manager.

- [ ] **Decouple CLI from Core Logic**:
    - **Goal**: Ensure files in `src/semantic_change/` are pure library modules without `if __name__ == "__main__":` blocks or `argparse` logic.
    - **Action**:
        - Remove CLI code from `src/semantic_change/embeddings_generation.py`.
        - Create dedicated runner scripts (e.g., `src/cli/run_batch.py`, `src/cli/run_ranking.py`) that import the logic and handle user interaction.
    - **Benefit**: Improves testability and strictly separates the "User Interface" (CLI) from the "Business Logic".


## IV. Testing & Quality Assurance

- [ ] **Implement Unit Testing Suite**:
    - Set up `pytest` as the primary testing framework.
    - Create a `tests/` directory to house test modules.
- [ ] **Core Logic Tests**:
    - **`corpus.py`**: Test SQLite connection, metadata retrieval, and sentence sampling.
    - **`ingestor.py`**: Test raw file processing, tokenization accuracy, and database population.
    - **`embedding.py`**: Test vector extraction (using small/mock models) and alignment logic.
    - **`wsi.py`**: Test clustering consistency with dummy data.
- [ ] **Integration Tests**:
    - Verify the end-to-end flow from raw text ingestion to embedding generation and basic analysis.



## V. Major Feature: Substitution-based WSI (Eyal et al., 2022)

Implement the graph-based Word Sense Induction approach based on "Large Scale Substitution-based Word Sense Induction", which uses MLM top-k substitutes instead of dense vectors for clustering.

### 1. MLM Substitute Generation (`src/semantic_change/embedding.py`)
- [ ] **Extend `BertEmbedder`**: Add a method `generate_substitutes(sentences, k=5)` that:
    - Tokenizes sentences and masks the target word.
    - Runs the MLM head to predict the top-k most probable *complete* words (handling BPE tokenization issues described in `workplan3.md`).
    - Returns a list of lists: `[[substitute1, substitute2, ...], ...]` for each sentence instance.

### 2. Graph Construction & Clustering (`src/semantic_change/wsi.py`)
- [ ] **Implement Graph Builder**:
    - Create a co-occurrence graph where nodes are substitute words.
    - Edges are weighted by how often two substitutes appear together in the top-k predictions for the same instance.
- [ ] **Implement Community Detection**:
    - Add `louvain` or `leiden` algorithm support (using `python-louvain` or `cdlib`).
    - The communities detected in this graph become the "senses".
- [ ] **Instance Tagging**:
    - Assign each original sentence to a sense cluster by calculating the Jaccard similarity between its instance-specific substitutes and the sense's representative words.

### 3. Integration & Visualization (`src/main.py`, `src/semantic_change/visualization.py`)
- [ ] **New Visualization**:
    - Create a graph visualization (using `networkx` + `plotly`) showing the community structure of the substitutes.
    - Nodes are words, edges reflect the distance, colors indicate the sense cluster.


## VI. Major Feature: Add more WSI methods and measures for semantic change detection

### 1. Add more measures for semantic change detection (Periti & Montanelli 2024)

**Form-based Measures (Section 4.1):**
- [ ] **Cosine Distance (CD)** [91]: Distance between word prototypes.
- [ ] **Inverted similarity over word prototype (PRT)** [73]: Inverse of cosine similarity.
- [ ] **Time-diff (TD)** [116]: Average difference of predicted time probabilities.
- [ ] **Average Pairwise Distance (APD)** [45]: Average distance between all pairs of embeddings from the two periods.
- [ ] **Average of Average Inner Distances (APD-OLD/NEW)** [81]: Estimates change as the average degree of polysemy.
- [ ] **Hausdorff Distance (HD)** [138]: Max-min distance between embedding sets.
- [ ] **Difference between Token Embedding Diversities (DIV)** [72]: Absolute difference between coefficients of variation.
- [ ] **Cosine Distance between Semantic Prototypes (PDIS)** [105]: CD between average embeddings of all sense prototypes.
- [ ] **Difference between Prototype Embedding Diversities (PDIV)** [105]: Extension of DIV using semantic prototypes.

**Sense-based Measures (Section 4.2):**
- [ ] **Maximum Novelty Score (MNS)** [28]: Max ratio of embeddings from one period in a cluster.
- [ ] **Maximum Square (MS)** [115]: Square difference of cluster distributions.
- [ ] **Jensen-Shannon Divergence (JSD)** [64]: Symmetrical KL divergence between cluster distributions.
- [ ] **Coefficient of Semantic Change (CSC)** [64]: Weighted difference of elements in clusters.
- [ ] **Cosine Distance between Cluster Distributions (CDCD)** [7]: CD between cluster distribution vectors.
- [ ] **Entropy Difference (ED)** [45]: Difference in entropy (uncertainty) of distributions.
- [ ] **Average Pairwise Distance between Sense Prototypes (APDP)** [66]: APD over pairs of sense prototypes.
- [ ] **Wasserstein Distance (WD)** [100]: Optimal transport cost to reconfigure cluster distribution.

### 2. Add more WSI methods
(numbers refer to the reference section in Periti, Montanelli 2024 in the subdir ./references) 
- [ ] Affinity Propagation (AP)
- [ ] Gaussian Mixture Models (GMMs) [118]
- [ ] agglomerative clustering (AGG) (e.g., Reference [7])
- [ ] DBSCAN (e.g., Reference [65])
- [ ] HDBSCAN (e.g., Reference [118]) [Check this against our implementation]
- [ ] Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH) (e.g., Reference [118])
- [ ] A Posteriori affinity Propagation (APP) (e.g., Reference [105])
- [ ] Incremental Affinity Propagation based on Nearest neighbor Assignment (IAPNA)


## VII. Smaller Open Tasks
- [ ] **Export**: Allow exporting the analysis results (cluster assignments, neighbor lists) as CSV/JSON.
- [ ] **Cleaning up the GUI**:
    - Separate Test gui and logic. Move the test parts in the gui (and the logic) for text ingest and embedding creation to a major page "Testing".
    - **Language Selection & Model Compatibility**: Add a dropdown for Language (EN, DE, etc.). Ensure the selected spaCy model matches the language to prevent runtime errors.
- [x] **GUI changes**: add time info for tasks like text ingest and embedding creation because they take long (similar to tqdm)
    - **Implemented**: Replaced custom `progress_callback` pattern with `tqdm_class` parameter across all long-running operations
    - Added `stqdm` dependency for Streamlit-compatible progress bars
    - Updated files: `embeddings_generation.py`, `ingestor.py`, `run_ingest.py`, `reporting.py`, `rank_semantic_change.py`, `gui_app.py`
    - CLI uses standard `tqdm`, GUI uses `stqdm` (passed as `tqdm_class=stqdm`)


## VIII. Future Tasks
At the moment the context for the computation of an embedding and also for the viz of a lemma occurrence is always a sentence. We want to change this:
- [ ] **Free definition of the context**: Allow to use as much of the text around the focus word for the computationn of the embedding as user wants. 
- [ ] **Ingest summaries of longer contexts**: Extend the context by adding more context, for example by adding more passage of the same author using the focus term. 
- [ ] **Attune the attention**: for the longer contexts use weight schemas for the attention information like decaying / upscaling with distance. 
- [ ]  Implement "Dynamic Context" 
- [ ]  *   **`ContextRetriever` Helper:** Write logic that checks the user-requested `context_size`. 
    *   If `size == "sentence"`, use the DB text.
    *   If `size > sentence`, use `file_offset_start` to read the raw file around the word.



