import re
from typing import Any, Dict, Iterator, Optional

from datasets import Dataset, load_dataset

from .task_processor import create_task_processor
from .yaml_parser import TaskConfig


class DatasetLoader:
    """Loads datasets based on task configuration."""

    def __init__(self, config: TaskConfig, config_file_path: str):
        self.config = config
        self.config_file_path = config_file_path
        self._dataset = None
        self._task_processor = create_task_processor(config, config_file_path)

    def load_dataset(self, limit: Optional[int] = None) -> Dataset:
        """Load the dataset for this task."""
        if self._dataset is None:
            self._dataset = self._load_raw_dataset()
            # Apply task-specific document processing
            self._dataset = self._task_processor.process_documents(self._dataset)

        dataset = self._dataset
        if limit is not None and limit > 0:
            # Take only the first 'limit' examples
            dataset = dataset.select(range(min(limit, len(dataset))))

        return dataset

    def _load_raw_dataset(self) -> Dataset:
        """Load the raw dataset from HuggingFace."""
        dataset_name = self.config.dataset_name
        dataset_path = self.config.dataset_path

        # Determine which split to use
        split = self._get_split()

        # Load dataset
        if dataset_name:
            dataset = load_dataset(dataset_path, dataset_name, split=split)
        else:
            dataset = load_dataset(dataset_path, split=split)

        return dataset

    def _get_split(self) -> str:
        """Determine which dataset split to use."""
        # Priority: test_split > validation_split > training_split > 'test'
        if self.config.test_split:
            return self.config.test_split
        elif self.config.validation_split:
            return self.config.validation_split
        elif self.config.training_split:
            return self.config.training_split
        else:
            return "test"  # Default to test split

    def get_examples(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Get examples from the dataset."""
        dataset = self.load_dataset(limit)
        for example in dataset:
            yield example


class TemplateProcessor:
    """Processes text templates with variables like {{question}}."""

    @staticmethod
    def process_template(template: str, example: Dict[str, Any]) -> str:
        """Process a template string with example data."""
        if not template:
            return ""

        # Handle Jinja2-style template variables like {{question}}
        result = template

        # Find all template variables
        template_vars = re.findall(r"\{\{([^}]+)\}\}", template)

        for var in template_vars:
            var = var.strip()

            # Handle conditional expressions like {{answer if answer is defined else target}}
            if " if " in var and " else " in var:
                result = TemplateProcessor._process_conditional(result, var, example)
            else:
                # Simple variable substitution
                value = TemplateProcessor._get_nested_value(example, var)
                if value is not None:
                    result = result.replace(f"{{{{{var}}}}}", str(value))

        return result

    @staticmethod
    def _process_conditional(
        template: str, conditional: str, example: Dict[str, Any]
    ) -> str:
        """Process conditional template expressions."""
        # Parse "value if condition else default"
        if_match = re.match(r"(.+?)\s+if\s+(.+?)\s+else\s+(.+)", conditional.strip())
        if not if_match:
            return template

        value_expr, condition, default_expr = if_match.groups()
        value_expr = value_expr.strip()
        condition = condition.strip()
        default_expr = default_expr.strip()

        # Check if condition is met (simple "field is defined" check)
        if condition.endswith(" is defined"):
            field = condition.replace(" is defined", "").strip()
            has_field = TemplateProcessor._get_nested_value(example, field) is not None

            if has_field:
                # Use the main value expression
                if value_expr.startswith("answer.split("):
                    # Handle complex expressions like answer.split('####')[-1].strip()
                    value = TemplateProcessor._process_complex_expression(
                        value_expr, example
                    )
                else:
                    value = TemplateProcessor._get_nested_value(example, value_expr)
            else:
                # Use the default expression
                value = TemplateProcessor._get_nested_value(example, default_expr)
        else:
            # Fallback for other conditions
            value = TemplateProcessor._get_nested_value(example, default_expr)

        if value is not None:
            return template.replace(f"{{{{{conditional}}}}}", str(value))

        return template

    @staticmethod
    def _process_complex_expression(
        expression: str, example: Dict[str, Any]
    ) -> Optional[str]:
        """Process complex expressions like answer.split('####')[-1].strip()."""
        try:
            # Handle answer.split('####')[-1].strip() type expressions
            if "answer.split(" in expression:
                # Extract the answer field
                answer = example.get("answer", "")
                if not answer:
                    return None

                # Parse the split operation
                split_match = re.search(
                    r"answer\.split\(['\"](.*?)['\"]\)\[(-?\d+)\]", expression
                )
                if split_match:
                    delimiter = split_match.group(1)
                    index = int(split_match.group(2))

                    parts = answer.split(delimiter)
                    if abs(index) <= len(parts):
                        result = parts[index]

                        # Apply .strip() if present
                        if ".strip()" in expression:
                            result = result.strip()

                        return result

            return None
        except:
            return None

    @staticmethod
    def _get_nested_value(data: Dict[str, Any], key: str) -> Optional[Any]:
        """Get a value from nested dictionary using dot notation."""
        try:
            current = data
            for part in key.split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
        except:
            return None
