"""
Dependency checking utilities for the Semantic Change Analysis toolkit.

This module provides functions to check and install required dependencies.
"""
import subprocess
import sys


def check_spacy_transformer_deps(model_name: str) -> tuple[bool, str]:
    """
    Check if transformer model dependencies are installed.

    For spaCy transformer models (ending in '_trf'), this checks and optionally
    installs the required spacy-curated-transformers package and downloads
    the model if needed.

    Args:
        model_name: The spaCy model name (e.g., 'en_core_web_trf')

    Returns:
        Tuple of (ready, message):
            - ready=True means dependencies are satisfied and we can proceed
            - ready=False means dependencies were installed and app needs restart,
              or installation failed
            - message contains status info or error details
    """
    needs_restart = False
    messages = []

    # Check if this is a transformer model
    if not model_name.endswith("_trf"):
        return True, ""

    # Check if spacy-curated-transformers is installed
    try:
        import spacy_curated_transformers  # noqa: F401
    except ImportError:
        messages.append("Installing spacy-curated-transformers...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "spacy-curated-transformers"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, f"Failed to install spacy-curated-transformers: {result.stderr}"
        needs_restart = True

    # Check if the model is downloaded
    try:
        import spacy
        spacy.util.get_package_path(model_name)
    except (ImportError, Exception):
        messages.append(f"Downloading spaCy model: {model_name}...")
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, f"Failed to download model {model_name}: {result.stderr}"
        needs_restart = True

    if needs_restart:
        return False, "Dependencies installed. Please restart the app to use transformer models."

    return True, ""
