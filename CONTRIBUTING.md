# Contributing to Semantic Change Analysis

Thank you for your interest in contributing! This project aims to provide a robust toolkit for analyzing semantic change using contextual embeddings.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/sild.git
    ```
3.  **Install dependencies** using `uv` (see `README.md`).

## Code Style

-   We follow standard PEP 8 guidelines for Python code.
-   Please use type hints (`typing` module) for function arguments and return values.
-   Ensure all new functions and classes have docstrings explaining their purpose, arguments, and return values.

## Development Workflow

1.  Create a new branch for your feature or bugfix:
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  Make your changes.
3.  Test your changes using the provided GUI (`uv run streamlit run gui.py`) or command-line scripts.
4.  Commit your changes with clear, descriptive messages.
5.  Push your branch and open a Pull Request.

## Project Structure

-   `gui.py`: The main entry point for the Streamlit interface.
-   `main.py`: Core logic for single-word analysis.
-   `run_batch_analysis.py`: Logic for batch processing.
-   `semantic_change/`: Package containing the core modules:
    -   `corpus.py`: SQLite-backed corpus management.
    -   `embedding.py`: BERT/Transformer embedding generation.
    -   `wsi.py`: Word Sense Induction algorithms (Clustering).
    -   `visualization.py`: Plotly-based plotting.
    -   `reporting.py`: Statistics and report generation.

## Reporting Issues

If you encounter bugs or have feature requests, please open an issue on the GitHub repository. Provide as much detail as possible, including steps to reproduce the issue.
