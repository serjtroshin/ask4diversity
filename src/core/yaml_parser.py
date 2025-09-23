from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# Custom YAML constructor to handle lm-eval specific tags
def function_constructor(loader, node):
    """Handle !function tags by returning a string representation."""
    return f"!function {loader.construct_scalar(node)}"


def ignore_unknown_constructor(loader, tag_suffix, node):
    """Handle unknown tags by returning None or a string representation."""
    if isinstance(node, yaml.ScalarNode):
        return f"!{tag_suffix} {loader.construct_scalar(node)}"
    elif isinstance(node, yaml.SequenceNode):
        return [loader.construct_object(child) for child in node.value]
    elif isinstance(node, yaml.MappingNode):
        return {
            loader.construct_object(key): loader.construct_object(value)
            for key, value in node.value
        }
    return None


# Create a custom YAML loader with our constructors
class LMEvalYamlLoader(yaml.SafeLoader):
    pass


# Register custom constructors on our custom loader
LMEvalYamlLoader.add_constructor("!function", function_constructor)
LMEvalYamlLoader.add_multi_constructor("!", ignore_unknown_constructor)


@dataclass
class TaskConfig:
    """Configuration for a task loaded from YAML."""

    # Basic task info
    task: str
    dataset_path: str
    dataset_name: Optional[str] = None
    tag: Optional[str] = None

    # Data splits
    training_split: Optional[str] = None
    validation_split: Optional[str] = None
    test_split: Optional[str] = None

    # Text generation
    doc_to_text: str = ""
    doc_to_target: str = ""
    doc_to_choice: Optional[List[str]] = None
    description: str = ""

    # Output and generation
    output_type: str = "generate_until"
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Few-shot configuration
    num_fewshot: int = 0
    fewshot_config: Optional[Dict[str, Any]] = None

    # Filtering and metrics
    filter_list: List[Dict[str, Any]] = field(default_factory=list)
    metric_list: List[Dict[str, Any]] = field(default_factory=list)

    # Other
    process_docs: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class YamlTaskParser:
    """Parser for task YAML files compatible with lm-evaluation-harness format."""

    def __init__(self, tasks_root: Union[str, Path]):
        self.tasks_root = Path(tasks_root)

    def load_task_config(self, task_name: str) -> TaskConfig:
        """Load a task configuration from YAML file."""
        task_file = self._find_task_file(task_name)
        if not task_file:
            raise FileNotFoundError(f"Task file not found for: {task_name}")

        with open(task_file, "r") as f:
            config_data = yaml.load(f, Loader=LMEvalYamlLoader)

        # Handle includes (like _gpqa_n_shot_yaml)
        if "include" in config_data:
            base_config = self._load_include(config_data["include"], task_file.parent)
            # Merge with current config, current config takes precedence
            merged_config = {**base_config, **config_data}
            config_data = merged_config

        # Set task name if not provided
        if "task" not in config_data:
            config_data["task"] = task_name

        return TaskConfig(**self._clean_config(config_data))

    def _find_task_file(self, task_name: str) -> Optional[Path]:
        """Find the YAML file for a given task name."""
        # Try direct match first
        for yaml_file in self.tasks_root.rglob("*.yaml"):
            if yaml_file.stem == task_name or yaml_file.name == f"{task_name}.yaml":
                return yaml_file

        # Try looking for task definition in file content
        for yaml_file in self.tasks_root.rglob("*.yaml"):
            try:
                with open(yaml_file, "r") as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict) and content.get("task") == task_name:
                        return yaml_file
            except:
                continue

        return None

    def _load_include(self, include_name: str, current_dir: Path) -> Dict[str, Any]:
        """Load an included YAML file."""
        # Look for include file in current directory first
        include_file = current_dir / include_name
        if not include_file.exists():
            # Try with .yaml extension
            include_file = current_dir / f"{include_name}.yaml"

        if not include_file.exists():
            # Search in parent directories
            for parent in current_dir.parents:
                include_file = parent / include_name
                if include_file.exists():
                    break
                include_file = parent / f"{include_name}.yaml"
                if include_file.exists():
                    break
            else:
                raise FileNotFoundError(f"Include file not found: {include_name}")

        with open(include_file, "r") as f:
            return yaml.load(f, Loader=LMEvalYamlLoader)

    def _clean_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize configuration data."""
        # Handle special YAML constructs - preserve function references for task processor
        if "process_docs" in config_data:
            if isinstance(config_data["process_docs"], str) and config_data[
                "process_docs"
            ].startswith("!function"):
                # Extract the function name from "!function utils.process_docs"
                func_ref = config_data["process_docs"].replace("!function ", "")
                config_data["process_docs"] = func_ref

        # Ensure required fields have defaults
        if "generation_kwargs" not in config_data:
            config_data["generation_kwargs"] = {}

        if "filter_list" not in config_data:
            config_data["filter_list"] = []

        if "metric_list" not in config_data:
            config_data["metric_list"] = []

        # Get valid fields for TaskConfig dataclass
        valid_fields = {f.name for f in TaskConfig.__dataclass_fields__.values()}

        # Check for unparsed fields and warn
        unparsed_fields = set(config_data.keys()) - valid_fields
        if unparsed_fields:
            print(
                f"Warning: The following fields from YAML are not parsed and will be ignored: {sorted(unparsed_fields)}"
            )

        # Clean out fields not in TaskConfig dataclass
        cleaned = {k: v for k, v in config_data.items() if k in valid_fields}

        return cleaned


def load_task_from_file(config_file_path: Union[str, Path]) -> TaskConfig:
    """Load a task configuration from an explicit YAML file path."""
    config_file = Path(config_file_path)

    if not config_file.exists():
        # print current dir
        print(f"Current directory: {Path.cwd()}")
        raise FileNotFoundError(
            f"Task configuration file not found: {config_file_path}"
        )

    with open(config_file, "r") as f:
        config_data = yaml.load(f, Loader=LMEvalYamlLoader)

    # Handle includes (like _gpqa_n_shot_yaml)
    if "include" in config_data:
        base_config = load_include_from_file(config_data["include"], config_file.parent)
        # Merge with current config, current config takes precedence
        merged_config = {**base_config, **config_data}
        config_data = merged_config

    # Set task name from file if not provided
    if "task" not in config_data:
        config_data["task"] = config_file.stem

    return TaskConfig(**clean_config_data(config_data))


def load_include_from_file(include_name: str, config_dir: Path) -> Dict[str, Any]:
    """Load an include file from the same directory as the config file."""
    include_file = config_dir / f"{include_name}.yaml"
    if not include_file.exists():
        include_file = config_dir / f"{include_name}"
        if not include_file.exists():
            raise FileNotFoundError(f"Include file not found: {include_name}")

    with open(include_file, "r") as f:
        return yaml.load(f, Loader=LMEvalYamlLoader)


def clean_config_data(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and normalize configuration data."""
    # Handle special YAML constructs - preserve function references for task processor
    if "process_docs" in config_data:
        if isinstance(config_data["process_docs"], str) and config_data[
            "process_docs"
        ].startswith("!function"):
            # Extract the function name from "!function utils.process_docs"
            func_ref = config_data["process_docs"].replace("!function ", "")
            config_data["process_docs"] = func_ref

    # Ensure required fields have defaults
    if "generation_kwargs" not in config_data:
        config_data["generation_kwargs"] = {}

    if "filter_list" not in config_data:
        config_data["filter_list"] = []

    if "metric_list" not in config_data:
        config_data["metric_list"] = []

    # Get valid fields for TaskConfig dataclass
    valid_fields = {f.name for f in TaskConfig.__dataclass_fields__.values()}

    # Check for unparsed fields and warn
    unparsed_fields = set(config_data.keys()) - valid_fields
    if unparsed_fields:
        print(
            f"Warning: The following fields from YAML are not parsed and will be ignored: {sorted(unparsed_fields)}"
        )

    # Clean out fields not in TaskConfig dataclass
    cleaned = {k: v for k, v in config_data.items() if k in valid_fields}

    return cleaned


def load_task(
    task_name: str, tasks_root: Optional[Union[str, Path]] = None
) -> TaskConfig:
    """Load a task configuration from task name (legacy function)."""
    if tasks_root is None:
        tasks_root = Path(__file__).parent.parent / "tasks"

    parser = YamlTaskParser(tasks_root)
    return parser.load_task_config(task_name)
