from setuptools import setup, find_packages

setup(
    name="semantic_change",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0,<2.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "torch>=2.0.0",
        "transformers>=4.48.0",
        "spacy>=3.6.0",
        "spacy-transformers>=1.2.0",
        "spacy-curated-transformers>=0.2.0",
        "chromadb>=0.4.0",
        "plotly>=5.15.0",
        "streamlit>=1.27.0",
        "umap-learn>=0.5.3",
        "hdbscan>=0.8.33",
        "tqdm>=4.65.0",
        "networkx>=3.1",
    ],
)