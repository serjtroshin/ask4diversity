from .evaluator import TaskEvaluator
from .sequential_evaluator import SequentialTaskEvaluator
from .model_wrapper import BaseModelWrapper


def choose_task_evaluator(
    task_type: str,
) -> TaskEvaluator:
    """Create a TaskEvaluator instance with the given parameters."""

    if task_type == "iteration":
        return SequentialTaskEvaluator  # iterative procedure
    elif task_type == "sequential" or task_type == "parallel":
        return TaskEvaluator  # normal generation
    else:
        raise ValueError(
            f"Unknown task type: {task_type}. Supported types are 'iteration', 'sequential', and 'parallel'."
        )
