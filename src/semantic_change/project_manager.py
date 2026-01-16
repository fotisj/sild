"""
Project metadata management for semantic change analysis.

A project represents a paired comparison of two text corpora (t1 and t2)
with their associated embeddings. Each project has:
- A unique 4-digit ID (internal, stable)
- User-given name and labels (can change)
- Paths to the SQLite corpus databases
"""
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Any


class ProjectManager:
    """Manages project metadata stored in a JSON file."""

    def __init__(self, storage_path: str = "data/projects.json"):
        self.storage_path = storage_path
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load projects from storage file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"projects": {}, "active_project": None}

    def _save(self) -> None:
        """Save projects to storage file."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2)

    def _generate_id(self) -> str:
        """Generate a unique 4-digit project ID."""
        existing_ids = set(self._data["projects"].keys())
        while True:
            new_id = f"{random.randint(1000, 9999)}"
            if new_id not in existing_ids:
                return new_id

    def create_project(
        self,
        name: str,
        label_t1: str,
        label_t2: str,
        db_t1: str = "data/corpus_t1.db",
        db_t2: str = "data/corpus_t2.db"
    ) -> str:
        """
        Create a new project.

        Args:
            name: User-friendly project name
            label_t1: Display label for first time period (e.g., "1800")
            label_t2: Display label for second time period (e.g., "1900")
            db_t1: Path to SQLite database for t1 corpus
            db_t2: Path to SQLite database for t2 corpus

        Returns:
            The generated project ID
        """
        project_id = self._generate_id()
        self._data["projects"][project_id] = {
            "name": name,
            "label_t1": label_t1,
            "label_t2": label_t2,
            "db_t1": db_t1,
            "db_t2": db_t2,
            "created": datetime.now().isoformat()
        }
        # Set as active if it's the first project
        if self._data["active_project"] is None:
            self._data["active_project"] = project_id
        self._save()
        return project_id

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project metadata by ID."""
        return self._data["projects"].get(project_id)

    def get_active_project(self) -> Optional[Dict[str, Any]]:
        """Get the currently active project with its ID included."""
        active_id = self._data.get("active_project")
        if active_id and active_id in self._data["projects"]:
            project = self._data["projects"][active_id].copy()
            project["id"] = active_id
            return project
        return None

    def get_active_project_id(self) -> Optional[str]:
        """Get the ID of the currently active project."""
        return self._data.get("active_project")

    def set_active_project(self, project_id: str) -> bool:
        """Set the active project by ID."""
        if project_id in self._data["projects"]:
            self._data["active_project"] = project_id
            self._save()
            return True
        return False

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects with their IDs."""
        result = []
        for pid, pdata in self._data["projects"].items():
            entry = pdata.copy()
            entry["id"] = pid
            result.append(entry)
        return result

    def update_project(self, project_id: str, **kwargs) -> bool:
        """Update project metadata fields."""
        if project_id not in self._data["projects"]:
            return False
        for key, value in kwargs.items():
            if key in ["name", "label_t1", "label_t2", "db_t1", "db_t2"]:
                self._data["projects"][project_id][key] = value
        self._save()
        return True

    def delete_project(self, project_id: str) -> bool:
        """Delete a project (metadata only, not the actual data)."""
        if project_id in self._data["projects"]:
            del self._data["projects"][project_id]
            if self._data["active_project"] == project_id:
                # Set another project as active, or None
                remaining = list(self._data["projects"].keys())
                self._data["active_project"] = remaining[0] if remaining else None
            self._save()
            return True
        return False

    def ensure_default_project(
        self,
        label_t1: str = "t1",
        label_t2: str = "t2",
        db_t1: str = "data/corpus_t1.db",
        db_t2: str = "data/corpus_t2.db"
    ) -> str:
        """
        Ensure a default project exists. Creates one if none exist.

        Returns the active project ID.
        """
        if not self._data["projects"]:
            return self.create_project(
                name="Default Project",
                label_t1=label_t1,
                label_t2=label_t2,
                db_t1=db_t1,
                db_t2=db_t2
            )
        return self._data["active_project"]


def get_collection_name(project_id: str, period: str, model_name: str) -> str:
    """
    Generate a standardized collection name for the vector store.

    Args:
        project_id: 4-digit project identifier
        period: "t1" or "t2"
        model_name: HuggingFace model name

    Returns:
        Collection name in format: embeddings_{project_id}_{period}_{safe_model}
    """
    safe_model = model_name.replace("/", "_").replace("-", "_")
    return f"embeddings_{project_id}_{period}_{safe_model}"
