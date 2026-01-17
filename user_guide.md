# User Guide: Semantic Change Analysis

This guide provides a detailed walkthrough of using the Semantic Change Analysis Toolkit.

## 1. Preparing Your Data

Before you can run any analysis, you need text data.

1.  **Acquire Text Corpora:** You need two datasets representing two different time periods (e.g., 1800-1850 and 1900-1950). These should be plain text files (`.txt`).
2.  **Organize Files:** Create two folders, one for each period, and place your text files inside them.
    *   Example: `data_source/t1/` and `data_source/t2/`.
3.  **Run Ingestion:** Use the command line to "ingest" this data. This converts the raw text into a searchable database.
    ```bash
    uv run python src/run_ingest.py --input-t1 data_source/t1 --input-t2 data_source/t2 --label-t1 1800 --label-t2 1900
    ```
    *   This process may take some time depending on the size of your data.
    *   It creates `corpus_t1.db` and `corpus_t2.db` in the `data/` folder.

## 2. Using the GUI

Launch the graphical interface:
```bash
uv run streamlit run gui.py
```

### Sidebar Configuration

*   **Navigation:** Switch between the four main modules of the application.
*   **Global Settings:**
    *   **Data Output Directory:** Point this to where your `.db` files are (default is `data`).
    *   **Save Configuration:** Persists your settings to `config.json` so they reload next time.
*   **Time Period Labels:** Set the display names for your two periods (e.g., "1800" and "1900").
*   **Database Status:** Shows ✅ if your corpus databases are found.

---


### Page 1: Data Ingestion (GUI Alternative)
If you prefer not to use the command line, you can perform ingestion here.

*   **Corpus Configuration:**
    *   **Input Directory:** Path to your raw text files for each period.
*   **File Encoding:** Select the encoding of your text files (usually UTF-8).
*   **Limit files (for testing):** Check this to process only a small random subset (e.g., 25 files) to quickly test the pipeline.
*   **SpaCy Model:** Select the language model for lemmatization (default: `en_core_web_sm`).
*   **Run Ingestion Process:** Starts the conversion. This may take a while.
*   **Danger Zone:**
    *   **Wipe SQLite Databases:** deletes `corpus_t1.db` and `corpus_t2.db`. Use with caution!

---


### Page 2: Embeddings & Models (Batch Analysis)
Use this to process *all* shared nouns between the two periods. This is **required** before you can see semantic change scores.

#### Tab: Create New Embeddings
*   **Select Model:** Choose a Hugging Face model (e.g., `bert-base-uncased`).
*   **Minimum Frequency:** Only process words that appear at least this many times in *both* corpora.
*   **Max Samples per Word:** How many sentences to extract per word per period (default: 200). Lower is faster; higher is more accurate.
*   **Custom Words:** Force specific words to be processed even if they are rare.
*   **Test mode (quick):** Runs a very small batch for debugging.
*   **Layer Selection:** (Advanced) Choose which Transformer layers to use. Default is the last layer (`-1`).
*   **Start Batch Process:** Begins the embedding generation. This runs on your CPU/GPU and saves vectors to `data/chroma_db`.

#### Tab: Manage Existing
*   **Delete:** Remove generated embeddings for a specific model to free up space.

---


### Page 3: Corpus Reports
*   **Database Status:** Summary of document and token counts.
*   **Top N Shared Words:** How many top frequent words to analyze in the report.
*   **Select Model for Semantic Change:** Choose which pre-computed embeddings to use for calculating shift scores.
*   **Generate Comparison Report:** Creates a Markdown report comparing word frequencies and semantic shift scores.

---


### Page 4: Analysis Dashboard (Single Word)
This is for deep-diving into a specific word.

**Parameters:**
1.  **Select Embedding Set:** Choose the pre-computed embeddings to use.
2.  **Model ID:** The Hugging Face model name (usually auto-filled).
3.  **Target Word:** Enter the word you want to study (e.g., "cell").
4.  **POS Filter:** Select "NOUN", "VERB", or "ADJ" to filter usages.
5.  **Samples per Period:** How many sentences to analyze (max is defined by what you generated in Batch Analysis).
6.  **Clustering Algorithm:**
    *   **HDBSCAN:** Good for finding natural clusters and ignoring noise.
    *   **K-Means:** Forces data into *k* clusters.
    *   **Spectral:** Graph-based clustering.
7.  **Dimensionality Reduction:**
    *   **Pre-clustering:** Reduce dimensions (e.g., via PCA) before clustering to improve performance.
    *   **Visualization:** Method to reduce data to 2D for the charts (PCA, UMAP, t-SNE).
8.  **Neighbors (k):** How many nearest semantic neighbors to find for each cluster.

**Actions:**
*   **Run Analysis:** Generates the plots.
*   **Archive Results:** Downloads all current plots as a ZIP file.

**Interpreting Results:**
*   **Time Period Clustering:** Shows usages colored by year. Separated colors = semantic shift.
*   **Sense Clusters:** Shows usages colored by their "sense" (meaning).
*   **Semantic Neighbors:** Graph showing the "centroid" of a cluster and its closest words. Helps you label the sense (e.g., "prison" vs. "biology" for "cell").

## 3. Command-Line Workflow

You can run the full pipeline from the terminal.

### Step 1: Batch Embedding
Generate embeddings for all frequent words.
```bash
uv run python -m src.semantic_change.embeddings_generation --model bert-base-uncased --max-samples 200
```

### Step 2: Rank Semantic Change
Calculate the shift for all shared words to find the most interesting ones.
```bash
uv run python src/rank_semantic_change.py --output output/ranking.csv
```
Open `output/ranking.csv` to see the words with the highest distance scores.

### Step 3: Single Word Analysis
Deep dive into a specific word (e.g., "factory").
```bash
uv run python main.py --word factory --model bert-base-uncased
```

## 4. HPC & GPU Cluster Support

For large datasets or complex models, you can offload the heavy lifting (ingestion and embedding generation) to a SLURM-based GPU cluster.

### Prerequisites
1.  SSH access to the cluster login node (e.g., `gwdu101`).
2.  SSH public key authentication configured (passwordless login).
3.  Apptainer (Singularity) installed on the cluster.

### A. Building the Container

You usually build the container **on the cluster** (since it requires Linux) or locally if you have WSL2.

**Option A: Build on Cluster (Recommended)**
1.  Push your code first (so the definition file exists there):
    ```bash
    # This works on Windows too (automatically falls back to tar+scp if rsync is missing)
    python -m src.cli.hpc push code
    ```
2.  SSH into the cluster and build:
    ```bash
    ssh your_user@gwdu101
    cd sild
    
    # Standard build command
    apptainer build --fakeroot sild.sif hpc/sild.def
    ```

    **Troubleshooting Build Errors:**
    If you see errors like `failed to create build parent dir` or `no space left on device`, you need to redirect the temporary files to your own directory:
    ```bash
    mkdir -p ~/apptainer_tmp ~/apptainer_cache
    
    # In one line:
    APPTAINER_TMPDIR=~/apptainer_tmp APPTAINER_CACHEDIR=~/apptainer_cache apptainer build --fakeroot sild.sif hpc/sild.def
    ```

**Option B: Build Locally (WSL2)**
If you have Linux/WSL2 with root access:
```bash
sudo apptainer build hpc/sild.sif hpc/sild.def
rsync -avz hpc/sild.sif your_user@gwdu101:sild/
```

### B. Remote Workflow

**1. Push Data & Code**
```bash
# Push code
uv run python -m src.cli.hpc push code

# Push raw data (e.g., your 't1' folder)
uv run python -m src.cli.hpc push data --path data_source/t1
uv run python -m src.cli.hpc push data --path data_source/t2
```

**2. Submit Jobs**
Use the CLI to generate and submit SLURM jobs automatically.

*   **Ingestion:**
    ```bash
    uv run python -m src.cli.hpc submit ingest \
      --input-t1 data_source/t1 \
      --input-t2 data_source/t2 \
      --label-t1 1800 \
      --label-t2 1900
    ```

*   **Embedding Generation:**
    ```bash
    uv run python -m src.cli.hpc submit embed --model bert-base-uncased --min-freq 50

    # You can submit multiple models
    uv run python -m src.cli.hpc submit embed --model LSX-UniWue/ModernGBERT_1B --min-freq 30
    ```

**3. Pull Results**
Once the jobs are finished (check with `squeue` via SSH), download the results.
```bash
# Pull the generated database
uv run python -m src.cli.hpc pull --remote-path data/corpus_t1.db --local-path data/
uv run python -m src.cli.hpc pull --remote-path data/corpus_t2.db --local-path data/

# Pull the vector store (ChromaDB)
uv run python -m src.cli.hpc pull --remote-path data/chroma_db --local-path data/
```