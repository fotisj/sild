# Semantic Change Analysis Toolkit

This toolkit provides a comprehensive pipeline for analyzing semantic change of words over time using contextualized word embeddings from Transformer models. It includes modules for corpus ingestion, embedding generation, word sense induction, and visualization. A user-friendly GUI allows for interactive configuration and analysis.

## Workflow Overview

The analysis process is divided into three main stages:

1.  **Data Ingestion (Model-Agnostic):** Raw text corpora from different time periods are processed into a structured format (SQLite database). This step involves tokenization and lemmatization. **This only needs to be run once per corpus.**

2.  **Batch Embedding & Ranking (Model-Specific):** 
    -   The system queries sentences from the ingested databases.
    -   It generates embeddings for frequent words using your chosen Transformer model.
    -   It then **ranks** these words by their semantic shift (cosine distance) to help you find interesting cases.

3.  **Detailed Analysis (Visualization):** 
    -   You select a specific word (e.g., one with a high shift score).
    -   The system clusters its embeddings to find distinct senses (WSI).
    -   Visualizations show how these senses evolve over time.

> **Key Takeaway:** You can experiment with different models (e.g., BERT, RoBERTa) on the same ingested data without ever needing to re-ingest the corpus.

## Installation for Non-Programmers

If you are new to Python, follow these steps to get started:

1.  **Install Python:**
    *   Go to [python.org](https://www.python.org/downloads/).
    *   Download the latest version for your operating system (Windows/macOS).
    *   Run the installer. **Important:** On Windows, make sure to check the box that says **"Add Python to PATH"** before clicking Install.

2.  **Open a Terminal:**
    *   **Windows:** Press the `Windows Key`, type `PowerShell`, and press Enter.
    *   **Mac/Linux:** Press `Cmd + Space`, type `Terminal`, and press Enter.

3.  **Install the Project Manager (`uv`):**
    *   In the terminal window, type the following command and press Enter:
        ```bash
        pip install uv
        ```
    *   This tool will help you install all other required software easily.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Dependencies:** This project uses `uv` for fast package management.
    ```bash
    pip install uv
    uv pip install -r requirements.txt
    ```

## Usage

### 1. Ingest Your Corpora

Place your raw text files for each time period into separate directories, for example:
- `data_source/t1/`
- `data_source/t2/`

Then, run the ingestion script:

```bash
uv run python src/run_ingest.py --input-t1 data_source/t1 --input-t2 data_source/t2 --label-t1 1800 --label-t2 1900
```
This will create `data/corpus_t1.db` and `data/corpus_t2.db`. You only need to do this once.

### 2. Launch the GUI

The easiest way to run an analysis is through the Streamlit-based GUI.

```bash
uv run streamlit run gui.py
```

This will launch a web interface (usually at `http://localhost:8501`) where you can:
-   **Configure Settings:** Set the data directory, embedding model (from Hugging Face), and analysis parameters.
-   **Run Single Word Analysis:** Interactively analyze a focus word, view its clusters, and see the nearest neighbors for each sense.
-   **Run Batch Analysis:** Pre-compute embeddings for all shared nouns between the two corpora.
-   **View Reports:** Generate and view a Markdown report comparing word frequencies.

### 3. Command-Line Usage (Advanced)

You can also run analyses directly from the command line:

**Step 1: Batch Embedding Generation**
Pre-compute embeddings for frequent words.
```bash
uv run python -m src.semantic_change.embeddings_generation --model bert-base-uncased --max-samples 200
```

**Step 2: Rank Semantic Change**
Calculate the shift for all shared words to find the most interesting ones.
```bash
uv run python src/rank_semantic_change.py --output output/ranking.csv
```

**Step 3: Single Word Analysis**
Deep dive into a specific word.
```bash
uv run python main.py --word factory --model bert-base-uncased
```

## HPC Integration

This project includes tools to run computationally intensive tasks (Ingestion, Embedding) on a SLURM-based HPC cluster.

See `user_guide.md` for detailed instructions on:
1.  Pushing code and data to the cluster (`src.cli.hpc push`).
2.  Submitting jobs (`src.cli.hpc submit`).
3.  Pulling results back to your local machine (`src.cli.hpc pull`).
