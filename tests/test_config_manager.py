"""
Tests for the config_manager module.
"""
import json
import os
import tempfile
import pytest

from semantic_change.config_manager import (
    AppConfig,
    load_config,
    save_config,
    get_db_paths,
    check_databases_exist,
)


class TestAppConfig:
    """Tests for the AppConfig dataclass."""

    def test_default_config_values(self):
        """Verify default values match expected configuration."""
        config = AppConfig()

        assert config.project_id == ""
        assert config.data_dir == "data"
        assert config.input_dir_t1 == "data_gutenberg/1800"
        assert config.input_dir_t2 == "data_gutenberg/1900"
        assert config.period_t1_label == "1800"
        assert config.period_t2_label == "1900"
        assert config.model_name == "bert-base-uncased"
        assert config.n_samples == 50
        assert config.min_cluster_size == 3
        assert config.wsi_algorithm == "hdbscan"
        assert config.layers == [-1]

    def test_load_missing_file(self):
        """Returns defaults when config file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "nonexistent_config.json")
            config = AppConfig.load(config_path)

            # Should return defaults
            assert config.model_name == "bert-base-uncased"
            assert config.data_dir == "data"

    def test_load_existing_file(self):
        """Loads and merges with defaults from existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")

            # Create a partial config file
            partial_config = {
                "model_name": "roberta-base",
                "n_samples": 100,
            }
            with open(config_path, "w") as f:
                json.dump(partial_config, f)

            config = AppConfig.load(config_path)

            # Check loaded values
            assert config.model_name == "roberta-base"
            assert config.n_samples == 100

            # Check defaults are applied for missing fields
            assert config.data_dir == "data"
            assert config.wsi_algorithm == "hdbscan"

    def test_save_and_reload(self):
        """Round-trip test: save and reload config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")

            # Create and save config
            original = AppConfig(
                project_id="1234",
                model_name="custom-model",
                n_samples=200,
            )
            original.save(config_path)

            # Reload and verify
            loaded = AppConfig.load(config_path)
            assert loaded.project_id == "1234"
            assert loaded.model_name == "custom-model"
            assert loaded.n_samples == 200

    def test_to_dict(self):
        """Verify to_dict returns correct dictionary."""
        config = AppConfig(project_id="test", model_name="my-model")
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["project_id"] == "test"
        assert d["model_name"] == "my-model"
        assert "data_dir" in d
        assert "wsi_algorithm" in d

    def test_get_db_paths(self):
        """Verify correct database path construction."""
        config = AppConfig(data_dir="custom_data")
        db_t1, db_t2 = config.get_db_paths()

        assert db_t1 == os.path.join("custom_data", "corpus_t1.db")
        assert db_t2 == os.path.join("custom_data", "corpus_t2.db")

    def test_check_databases_exist_true(self):
        """Returns True when both databases exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(data_dir=tmpdir)

            # Create both database files
            db_t1, db_t2 = config.get_db_paths()
            open(db_t1, "w").close()
            open(db_t2, "w").close()

            assert config.check_databases_exist() is True

    def test_check_databases_exist_false(self):
        """Returns False when databases don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(data_dir=tmpdir)
            assert config.check_databases_exist() is False

    def test_check_databases_exist_partial(self):
        """Returns False when only one database exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(data_dir=tmpdir)

            # Create only one database file
            db_t1, _ = config.get_db_paths()
            open(db_t1, "w").close()

            assert config.check_databases_exist() is False


class TestLegacyFunctions:
    """Tests for backward-compatible legacy functions."""

    def test_load_config_legacy(self):
        """Legacy load_config returns dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a temp config file
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"model_name": "test-model"}, f)

            # Test AppConfig.load directly with path
            config = AppConfig.load(config_path).to_dict()
            assert isinstance(config, dict)
            assert config["model_name"] == "test-model"

    def test_save_config_legacy(self):
        """Legacy save_config saves dict correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")

            # Create and save config
            app_config = AppConfig(model_name="saved-model", n_samples=75)
            app_config.save(config_path)

            with open(config_path) as f:
                loaded = json.load(f)
            assert loaded["model_name"] == "saved-model"
            assert loaded["n_samples"] == 75

    def test_get_db_paths_legacy(self):
        """Legacy get_db_paths returns correct paths."""
        config = {"data_dir": "my_data"}
        db_t1, db_t2 = get_db_paths(config)

        assert db_t1 == os.path.join("my_data", "corpus_t1.db")
        assert db_t2 == os.path.join("my_data", "corpus_t2.db")

    def test_check_databases_exist_legacy(self):
        """Legacy check_databases_exist works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_t1 = os.path.join(tmpdir, "db1.db")
            db_t2 = os.path.join(tmpdir, "db2.db")

            # Neither exists
            assert check_databases_exist(db_t1, db_t2) is False

            # Both exist
            open(db_t1, "w").close()
            open(db_t2, "w").close()
            assert check_databases_exist(db_t1, db_t2) is True
