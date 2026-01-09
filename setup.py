from setuptools import setup, find_packages

setup(
    name="semantic_change",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "scikit-learn",
        "matplotlib",
        "syntok",
        "numpy",
        "plotly",
        "pandas",
        "jupyterlab",
        "ipywidgets",
    ],
)
