"""
Configuration management for the Semantic Change Analysis toolkit.

This module provides centralized configuration handling using a dataclass pattern.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List

CONFIG_FILE = "config.json"


@dataclass
class AppConfig:
    """Application configuration with sensible defaults."""

    # Project identification
    project_id: str = ""  # Will be auto-generated if empty

    # Data directories
    data_dir: str = "data"
    input_dir_t1: str = "data_gutenberg/1800"
    input_dir_t2: str = "data_gutenberg/1900"

    # Period labels
    period_t1_label: str = "1800"
    period_t2_label: str = "1900"

    # Ingestion settings
    file_encoding: str = "utf-8"
    max_files: Optional[int] = None
    spacy_model: str = "en_core_web_sm"

    # Model settings
    model_name: str = "bert-base-uncased"
    layers: List[int] = field(default_factory=lambda: [-1])
    layer_op: str = "mean"
    lang: str = "en"
    pooling_strategy: str = "mean"

    # Analysis parameters
    target_word: str = "current"
    k_neighbors: int = 10
    min_cluster_size: int = 3
    n_clusters: int = 3
    wsi_algorithm: str = "hdbscan"
    pos_filter: str = "None"
    context_window: int = 0
    exact_match: bool = False

    # Pre-clustering reduction settings
    clustering_reduction: str = "None"
    clustering_n_components: int = 50
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    tsne_perplexity: int = 30

    # Visualization settings
    viz_reduction: str = "pca"
    use_umap: bool = True
    umap_n_components: int = 50

    # Sampling settings
    n_samples: int = 50
    viz_max_instances: int = 100
    min_freq: int = 25

    @classmethod
    def load(cls, path: str = CONFIG_FILE) -> "AppConfig":
        """
        Load config from file, using defaults for missing fields.

        Args:
            path: Path to the configuration JSON file

        Returns:
            AppConfig instance with loaded values merged with defaults
        """
        if os.path.exists(path):
            with open(path, "r") as f:
                loaded = json.load(f)

            # Get default values
            defaults = cls()
            default_dict = asdict(defaults)

            # Merge loaded values with defaults (loaded takes precedence)
            merged = {**default_dict, **loaded}

            # Handle special case for layers (ensure it's a list)
            if "layers" in merged and not isinstance(merged["layers"], list):
                merged["layers"] = [merged["layers"]]

            return cls(**merged)

        return cls()

    def save(self, path: str = CONFIG_FILE) -> None:
        """
        Save config to file.

        Args:
            path: Path to save the configuration JSON file
        """
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    def to_dict(self) -> dict:
        """Convert to dict for backward compatibility with existing code."""
        return asdict(self)

    def get_db_paths(self) -> tuple[str, str]:
        """
        Returns paths to both corpus databases.

        Returns:
            Tuple of (db_t1_path, db_t2_path)
        """
        db_t1 = os.path.join(self.data_dir, "corpus_t1.db")
        db_t2 = os.path.join(self.data_dir, "corpus_t2.db")
        return db_t1, db_t2

    def check_databases_exist(self) -> bool:
        """
        Check if both corpus databases exist.

        Returns:
            True if both databases exist, False otherwise
        """
        db_t1, db_t2 = self.get_db_paths()
        return os.path.exists(db_t1) and os.path.exists(db_t2)

    def ensure_project_id(self) -> str:
        """
        Ensure config has a valid project_id, creating one if necessary.

        Updates the config and saves to file if a new project_id is created.

        Returns:
            The project_id (existing or newly created)
        """
        if self.project_id:
            return self.project_id

        from semantic_change.project_manager import ProjectManager
        pm = ProjectManager()
        db_t1, db_t2 = self.get_db_paths()
        project_id = pm.ensure_default_project(
            label_t1=self.period_t1_label,
            label_t2=self.period_t2_label,
            db_t1=db_t1,
            db_t2=db_t2
        )
        self.project_id = project_id
        self.save()
        return project_id


def load_config() -> dict:
    """
    Legacy function for backward compatibility.
    Loads configuration from file, merging with defaults.

    Returns:
        Configuration dictionary
    """
    return AppConfig.load().to_dict()


def save_config(config: dict) -> None:
    """
    Legacy function for backward compatibility.
    Saves configuration to file.

    Args:
        config: Configuration dictionary to save
    """
    # Create AppConfig from dict (handles missing fields with defaults)
    defaults = AppConfig()
    default_dict = asdict(defaults)
    merged = {**default_dict, **config}

    # Handle layers field
    if "layers" in merged and not isinstance(merged["layers"], list):
        merged["layers"] = [merged["layers"]]

    app_config = AppConfig(**merged)
    app_config.save()


def get_db_paths(config: dict) -> tuple[str, str]:
    """
    Legacy function for backward compatibility.
    Returns the paths to both corpus databases.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (db_t1_path, db_t2_path)
    """
    db_t1 = os.path.join(config["data_dir"], "corpus_t1.db")
    db_t2 = os.path.join(config["data_dir"], "corpus_t2.db")
    return db_t1, db_t2


def check_databases_exist(db_t1: str, db_t2: str) -> bool:
    """
    Legacy function for backward compatibility.
    Checks if both corpus databases exist.

    Args:
        db_t1: Path to the first database
        db_t2: Path to the second database

    Returns:
        True if both databases exist, False otherwise
    """
    return os.path.exists(db_t1) and os.path.exists(db_t2)


def ensure_project_id(config: dict) -> str:
    """
    Legacy function for backward compatibility.
    Ensures config has a valid project_id, creating one if necessary.

    Updates the config dict in place and saves to file.

    Args:
        config: Configuration dictionary (modified in place)

    Returns:
        The project_id
    """
    if config.get("project_id"):
        return config["project_id"]

    from semantic_change.project_manager import ProjectManager
    pm = ProjectManager()
    db_t1, db_t2 = get_db_paths(config)
    project_id = pm.ensure_default_project(
        label_t1=config.get("period_t1_label", "t1"),
        label_t2=config.get("period_t2_label", "t2"),
        db_t1=db_t1,
        db_t2=db_t2
    )
    config["project_id"] = project_id
    save_config(config)
    return project_id
