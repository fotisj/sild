# User Guide: Semantic Change Analysis

This guide provides a detailed walkthrough of using the Semantic Change Analysis Toolkit.

## 1. Preparing Your Data

Before you can run any analysis, you need text data.

1.  **Acquire Text Corpora:** You need two datasets representing two different time periods (e.g., 1800-1850 and 1900-1950). These should be plain text files (`.txt`).
2.  **Organize Files:** Create two folders, one for each period, and place your text files inside them.
    *   Example: `my_data/corpus_1800/` and `my_data/corpus_1900/`.
3.  **Run Ingestion:** Use the command line to "ingest" this data. This converts the raw text into a searchable database.
    ```bash
    python run_ingest.py --input-dir-1800 my_data/corpus_1800 --input-dir-1900 my_data/corpus_1900
    ```
    *   This process may take some time depending on the size of your data.
    *   It creates `corpus_1800.db` and `corpus_1900.db` in the `data/` folder.

## 2. Using the GUI

Launch the graphical interface:
```bash
uv run streamlit run gui.py
```

### Configuration (Sidebar)
*   **Data Directory:** Point this to where your `.db` files are (default is `data`).
*   **Embedding Model:** Enter the name of a Hugging Face model (e.g., `bert-base-uncased`, `roberta-base`, `answerdotai/ModernBERT-base`). The first time you use a new model, it will be downloaded automatically.

### Tab 1: Analysis Dashboard (Single Word)
This is for deep-diving into a specific word.

1.  **Target Word:** Enter the word you want to study (e.g., "cell").
2.  **POS Filter:** Select "NOUN", "VERB", or "ADJ" to filter usages. For example, selecting "NOUN" for "watch" will ignore the verb "to watch".
3.  **Samples per Period:** How many random sentences to fetch from each time period. Higher numbers give better results but take longer.
4.  **Clustering Algorithm:**
    *   **HDBSCAN:** Good for finding natural clusters and ignoring noise (outliers).
    *   **K-Means:** Forces data into *k* clusters. Good if you know how many senses to expect.
    *   **Spectral:** Graph-based clustering, often good for complex shapes.
    *   **Agglomerative:** Hierarchical clustering.
5.  **Use UMAP Reduction:** Recommended. Reduces the complex 768-dimension vectors to 50 dimensions before clustering, which often improves accuracy.
6.  **Run Analysis:** Click the button to start.

**Interpreting Results:**
*   **Time Period Clustering:** Shows usages colored by year. If the colors are separated, the word's usage has likely changed.
*   **Sense Clusters:** Shows usages colored by their "sense" (meaning). Hover over points to read the sentence.
*   **Semantic Neighbors:** Shows the "centroid" (average meaning) of a cluster and its closest words in the vocabulary. This helps you label the sense (e.g., "prison", "biology" for the word "cell").

### Tab 2: Embeddings & Models (Batch Analysis)
Use this to process *all* shared nouns between the two periods.

1.  **Generate Embeddings:** Select a model and set criteria (e.g., Min Frequency).
2.  **Start Batch Process:** This will compute embeddings for all frequent words and save them to disk. This is **required** before you can see semantic change scores in the Corpus Reports.

### Tab 3: Corpus Reports
1.  **Generate Comparison Report:** Click this to see a table comparing word frequencies between the two time periods.
2.  **Semantic Change:** If you have run the Batch Analysis (Tab 2), this report will also show the "Semantic Change" score (Cosine Distance) for each word, allowing you to easily spot which words have shifted the most.

## 3. Command-Line Workflow

You can run the full pipeline from the terminal.

### Step 1: Batch Embedding
Generate embeddings for all frequent words.
```bash
python -m src.semantic_change.embeddings_generation --model bert-base-uncased --max-samples 200
```

### Step 2: Semantic Change Ranking
Calculate the shift for all shared words to find the most interesting ones.
```bash
python src/rank_semantic_change.py --output output/ranking.csv
```
Open `output/ranking.csv` to see the words with the highest distance scores.

### Step 3: Single Word Analysis
Deep dive into a specific word (e.g., "factory").
```bash
python main.py --word factory --model bert-base-uncased
```

## 4. Advanced Tips

*   **Model Choice:** `bert-base-uncased` is a standard baseline. `roberta-base` often performs better. Newer models like `ModernBERT` can be very efficient.
*   **Performance:** If the app is slow, try reducing "Samples per Period".
*   **Cluster Tuning:** If HDBSCAN returns too much "noise" (Cluster -1), try lowering "Min Cluster Size". If K-Means splits a single sense into two, try reducing *k*.