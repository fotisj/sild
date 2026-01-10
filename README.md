# Semantic Change Analysis Toolkit

This toolkit provides a comprehensive pipeline for analyzing semantic change of words over time using contextualized word embeddings from Transformer models. It includes modules for corpus ingestion, embedding generation, word sense induction, and visualization. A user-friendly GUI allows for interactive configuration and analysis.

## Workflow Overview

The analysis process is divided into two main stages:

1.  **Data Ingestion (Model-Agnostic):** Raw text corpora from different time periods are processed into a structured format (SQLite database). This step involves tokenization and lemmatization. **This only needs to be run once per corpus.** The generated databases are independent of any specific embedding model.

2.  **Analysis (Model-Specific):** This is where you use the GUI or scripts to analyze words.
    -   The system queries sentences from the ingested databases.
    -   It then loads your chosen Transformer model (e.g., `bert-base-uncased`, `answerdotai/ModernBERT-base`) on the fly.
    -   Embeddings are generated at this stage using the selected model.
    -   Finally, clustering (WSI) and visualization are performed.

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

3.  **Download spaCy Model:** The ingestion and neighbor filtering steps use a spaCy model.
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

### 1. Ingest Your Corpora

Place your raw text files for each time period into separate directories, for example:
- `data_gutenberg/1800/`
- `data_gutenberg/1900/`

Then, run the ingestion script:

```bash
python run_ingest.py --input-dir-1800 data_gutenberg/1800 --input-dir-1900 data_gutenberg/1900
```
This will create `data/ingested_1800.db` and `data/ingested_1900.db`. You only need to do this once.

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

**Single Word Analysis:**
```bash
python main.py --word factory --model bert-base-uncased
```

**Batch Analysis:**
```bash
python run_batch_analysis.py --model roberta-base --limit 1000
```
