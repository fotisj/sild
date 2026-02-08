"""
Tests for the vector_store module.
"""
import pytest
from unittest.mock import MagicMock, patch

from semantic_change.vector_store import VectorStore


class TestDeleteModelEmbeddings:
    """Tests for the delete_model_embeddings method."""

    def test_delete_model_embeddings_both_exist(self):
        """Successfully deletes both t1 and t2 collections."""
        with patch('chromadb.PersistentClient') as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            store = VectorStore(persistence_path="test/path")

            # Mock count to return values before deletion
            mock_coll = MagicMock()
            mock_coll.count.side_effect = [100, 80]
            store._collection_cache = {}

            # Mock get_or_create_collection for count calls
            store.get_or_create_collection = MagicMock(return_value=mock_coll)

            # Mock delete_collection to succeed
            store.delete_collection = MagicMock(side_effect=[
                (True, "Deleted: embeddings_1234_t1_bert_base_uncased"),
                (True, "Deleted: embeddings_1234_t2_bert_base_uncased"),
            ])

            # Mock count method
            store.count = MagicMock(side_effect=[100, 80])

            success, message, count_t1, count_t2 = store.delete_model_embeddings(
                "1234", "bert-base-uncased"
            )

            assert success is True
            assert count_t1 == 100
            assert count_t2 == 80
            assert "Deleted" in message

    def test_delete_model_embeddings_none_exist(self):
        """Returns appropriate message when no collections exist."""
        with patch('chromadb.PersistentClient') as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            store = VectorStore(persistence_path="test/path")

            # Mock count to raise exceptions (collections don't exist)
            store.count = MagicMock(side_effect=Exception("not found"))

            # Mock delete_collection to return not found
            store.delete_collection = MagicMock(side_effect=[
                (False, "embeddings_1234_t1_bert not found"),
                (False, "embeddings_1234_t2_bert not found"),
            ])

            success, message, count_t1, count_t2 = store.delete_model_embeddings(
                "1234", "bert"
            )

            assert success is False
            assert count_t1 == 0
            assert count_t2 == 0
            assert "not found" in message

    def test_delete_model_embeddings_partial(self):
        """Returns success when at least one collection is deleted."""
        with patch('chromadb.PersistentClient') as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            store = VectorStore(persistence_path="test/path")

            # Mock count - first exists, second doesn't
            store.count = MagicMock(side_effect=[50, Exception("not found")])

            # Mock delete_collection - first succeeds, second fails
            store.delete_collection = MagicMock(side_effect=[
                (True, "Deleted: embeddings_1234_t1_model"),
                (False, "embeddings_1234_t2_model not found"),
            ])

            success, message, count_t1, count_t2 = store.delete_model_embeddings(
                "1234", "model"
            )

            assert success is True  # At least one was deleted
            assert count_t1 == 50
            assert count_t2 == 0

    def test_delete_model_embeddings_model_name_sanitization(self):
        """Verifies model name is properly sanitized for collection names."""
        with patch('chromadb.PersistentClient') as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            store = VectorStore(persistence_path="test/path")
            store.count = MagicMock(return_value=0)
            store.delete_collection = MagicMock(return_value=(False, "not found"))

            # Test with model name containing slashes and hyphens
            store.delete_model_embeddings("1234", "org/model-name-v2")

            # Verify delete_collection was called with sanitized names
            calls = store.delete_collection.call_args_list
            assert len(calls) == 2

            # Collection names should have / and - replaced with _
            call1_name = calls[0][0][0]
            call2_name = calls[1][0][0]

            assert "org_model_name_v2" in call1_name
            assert "org_model_name_v2" in call2_name
            assert "/" not in call1_name
            assert "-" not in call1_name


class TestVectorStoreBasics:
    """Basic tests for VectorStore initialization."""

    def test_init_creates_client(self):
        """Verify PersistentClient is created with correct path."""
        with patch('chromadb.PersistentClient') as MockClient:
            store = VectorStore(persistence_path="custom/db/path")

            MockClient.assert_called_once_with(path="custom/db/path")
            assert store.persistence_path == "custom/db/path"

    def test_collection_cache_initialized(self):
        """Verify collection cache is initialized as empty dict."""
        with patch('chromadb.PersistentClient'):
            store = VectorStore()
            assert store._collection_cache == {}
