"""
Task-specific document processing module.
This module dynamically imports and applies process_docs functions from task directories.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from datasets import Dataset

from .yaml_parser import TaskConfig


class TaskProcessor:
    """Handles task-specific document processing by dynamically importing process_docs functions."""

    def __init__(self, config: TaskConfig, config_file_path: str):
        self.config = config
        self.config_file_path = Path(config_file_path)
        self._process_docs_fn = None

    def process_documents(self, dataset: Dataset) -> Dataset:
        """Process documents using the task-specific process_docs function if available."""
        process_fn = self._get_process_docs_function()

        if process_fn:
            try:
                return process_fn(dataset)
            except Exception as e:
                print(f"Warning: Task-specific processing failed: {e}")
                print("Returning unprocessed dataset...")
                return dataset
        else:
            # No processing function found, return dataset as-is
            return dataset

    def _get_process_docs_function(self) -> Optional[Callable]:
        """Dynamically import and return the process_docs function for this task."""
        if self._process_docs_fn is not None:
            return self._process_docs_fn

        # Check if task config specifies process_docs
        if not hasattr(self.config, "process_docs") or not self.config.process_docs:
            return None

        # Check if process_docs is a function reference like "utils.process_docs"
        if (
            isinstance(self.config.process_docs, str)
            and "." in self.config.process_docs
        ):
            module_name, func_name = self.config.process_docs.rsplit(".", 1)
            if module_name != "utils" or func_name != "process_docs":
                print(
                    f"Warning: Only 'utils.process_docs' is supported, got: {self.config.process_docs}"
                )
                return None

        # Try to find and import the utils module from the same directory as config file
        utils_module = self._find_and_import_utils_module()
        if utils_module and hasattr(utils_module, "process_docs"):
            self._process_docs_fn = utils_module.process_docs
            return self._process_docs_fn

        return None

    def _find_and_import_utils_module(self) -> Optional[Any]:
        """Find and import the utils module from the same directory as the config file."""
        # Look for utils.py in the same directory as the config file
        config_dir = self.config_file_path.parent
        utils_file = config_dir / "utils.py"

        if utils_file.exists():
            try:
                # Add the config directory to sys.path temporarily
                str_path = str(config_dir)
                if str_path not in sys.path:
                    sys.path.insert(0, str_path)

                # Import utils module
                spec = importlib.util.spec_from_file_location("utils", utils_file)
                utils_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(utils_module)

                # Remove from sys.path
                if str_path in sys.path:
                    sys.path.remove(str_path)

                return utils_module

            except Exception as e:
                print(f"Warning: Could not import utils from {utils_file}: {e}")
                return None

        print(f"Warning: No utils.py found in {config_dir}")
        return None


def create_task_processor(config: TaskConfig, config_file_path: str) -> TaskProcessor:
    """Create a task processor for the given configuration."""
    return TaskProcessor(config, config_file_path)
