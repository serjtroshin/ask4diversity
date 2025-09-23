import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from lm_eval.api.metrics import exact_match_hf_evaluate

# Import lm-eval's metric registry and functions
from lm_eval.api.registry import get_metric
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.utils import hash_string


class MetricCalculator:
    """Calculates evaluation metrics using lm-eval's metric functions."""

    def __init__(self, metric_configs: List[Dict[str, Any]]):
        self.metric_configs = metric_configs
        self.metric_functions = {}
        self.metric_kwargs = {}

        # Initialize metric functions
        for metric_config in metric_configs:
            metric_name = metric_config.get("metric", "unknown")

            try:
                # Get metric from lm-eval registry
                metric_fn = get_metric(metric_name)
                self.metric_functions[metric_name] = metric_fn

                # Extract kwargs for the metric (excluding standard keys)
                kwargs = {
                    key: metric_config[key]
                    for key in metric_config
                    if key not in ["metric", "aggregation", "higher_is_better"]
                }
                self.metric_kwargs[metric_name] = kwargs

            except Exception as e:
                raise ValueError(
                    f"Could not load metric '{metric_name}' from lm-eval: {e}"
                )

    def calculate_metrics(
        self, predictions: List[str], targets: List[str]
    ) -> Dict[str, float]:
        """Calculate all configured metrics using lm-eval's functions."""
        results = {}

        for metric_config in self.metric_configs:
            metric_name = metric_config.get("metric", "unknown")
            metric_fn = self.metric_functions[metric_name]
            kwargs = self.metric_kwargs[metric_name]

            try:
                # Handle different metric function signatures
                if metric_name == "exact_match":
                    # Use lm-eval's exact_match function directly
                    result = exact_match_hf_evaluate(
                        predictions=predictions, references=targets, **kwargs
                    )
                    score = result["exact_match"]
                elif metric_name == "acc":
                    # Calculate accuracy: proportion of predictions that match targets
                    correct = sum(
                        1
                        for pred, target in zip(predictions, targets)
                        if pred.strip() == target.strip()
                    )
                    score = correct / len(predictions) if len(predictions) > 0 else 0.0
                elif metric_name == "acc_norm":
                    # For simple prediction/target pairs, normalized accuracy is the same as accuracy
                    # In multiple choice contexts, this would be normalized by number of choices
                    correct = sum(
                        1
                        for pred, target in zip(predictions, targets)
                        if pred.strip() == target.strip()
                    )
                    score = correct / len(predictions) if len(predictions) > 0 else 0.0
                else:
                    # For other metrics, try calling directly
                    # Note: Some metrics may need different handling
                    if hasattr(metric_fn, "__call__"):
                        result = metric_fn(
                            predictions=predictions, references=targets, **kwargs
                        )
                        if isinstance(result, dict) and metric_name in result:
                            score = result[metric_name]
                        elif isinstance(result, (int, float)):
                            score = result
                        else:
                            score = (
                                np.mean(result)
                                if hasattr(result, "__iter__")
                                else result
                            )
                    else:
                        raise ValueError(
                            f"Metric '{metric_name}' does not have a valid implementation"
                        )

                results[metric_name] = score

            except Exception as e:
                raise ValueError(
                    f"Error calculating metric '{metric_name}' with lm-eval: {e}"
                )

        return results


class LMEvalResultsLogger:
    """Enhanced results logger using lm-eval's EvaluationTracker for standardized saving."""

    def __init__(
        self,
        output_path: str,
        model_name: str = "transformers_model",
        task_name: str = "evaluation_task",
        use_lmeval_format: bool = True,
    ):
        self.output_path = output_path
        self.model_name = model_name
        self.task_name = task_name
        self.use_lmeval_format = use_lmeval_format

        # Sample storage in lm-eval format
        self.samples = []
        self.start_time = time.perf_counter()

        # Initialize lm-eval's EvaluationTracker if using lm-eval format
        if self.use_lmeval_format:
            self.evaluation_tracker = EvaluationTracker(output_path=output_path)
            # Set model name for proper directory structure
            self.evaluation_tracker.general_config_tracker.model_name = model_name
            self.evaluation_tracker.general_config_tracker.model_name_sanitized = (
                self._sanitize_model_name(model_name)
            )

    def _sanitize_model_name(self, name: str) -> str:
        """Sanitize model name for file system compatibility."""
        return re.sub(r"[^\w\-_.]", "_", name)

    def log_sample(
        self,
        example: Dict[str, Any],
        prompt: str,
        generated_text: str,
        filtered_answer: str,
        expected_answer: str,
        is_correct: bool,
        doc_id: int = None,
        iteration_id: int = None,  # optionally log iteration ID
    ):
        """Log a single sample result in lm-eval format."""

        if self.use_lmeval_format:
            # Format sample according to lm-eval structure
            sample = {
                "doc_id": doc_id if doc_id is not None else len(self.samples),
                "doc": example,
                "target": expected_answer,
                "arguments": [[prompt]],  # lm-eval expects list of arguments
                "resps": [[generated_text]],  # lm-eval expects list of responses
                "filtered_resps": [
                    [filtered_answer]
                ],  # lm-eval expects filtered responses
                "filter": "none",  # default filter name
                "metrics": ["exact_match"],  # metrics applied
                "exact_match": 1.0 if is_correct else 0.0,  # metric result
                # Hashes for integrity checking (lm-eval standard)
                "doc_hash": hash_string(
                    json.dumps(example, default=str, sort_keys=True)
                ),
                "prompt_hash": hash_string(prompt),
                "target_hash": hash_string(str(expected_answer)),
            }
        else:
            # Simple format
            doc_id = doc_id if doc_id is not None else len(self.samples)
            sample = {
                "example": example,
                "prompt": prompt,
                "generated_text": generated_text,
                "filtered_answer": filtered_answer,
                "expected_answer": expected_answer,
                "is_correct": is_correct,
                "doc_id": doc_id if doc_id is not None else len(self.samples),
            }
        if iteration_id is not None:
            sample["iteration_id"] = iteration_id
        else:
            sample["iteration_id"] = 0  # keep the same dataset structure

        self.samples.append(sample)

    def log_final_metrics(self, metrics: Dict[str, float]):
        """Log final aggregated metrics."""
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)

        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.4f}")

        print(f"\nTotal samples: {len(self.samples)}")
        print("=" * 60)

    def save_results(
        self, metrics: Dict[str, float], model_args: Dict[str, Any] = None
    ):
        """Save results using lm-eval's format and functionality."""

        if self.use_lmeval_format:
            self._save_lmeval_format(metrics, model_args)
        else:
            self._save_simple_format(metrics)

    def _save_lmeval_format(
        self, metrics: Dict[str, float], model_args: Dict[str, Any] = None
    ):
        """Save results using lm-eval's EvaluationTracker."""

        # Calculate aggregated metrics in lm-eval format
        total_samples = len(self.samples)

        # Build results structure following lm-eval schema
        results_dict = {
            "results": {
                self.task_name: {
                    **{
                        f"{metric_name},none": score
                        for metric_name, score in metrics.items()
                    },
                    "alias": self.task_name,
                    "samples": total_samples,
                }
            },
            "group_subtasks": {},
            "configs": {
                self.task_name: {
                    "task": self.task_name,
                    "metric_list": [
                        {"metric": metric_name} for metric_name in metrics.keys()
                    ],
                }
            },
            "versions": {self.task_name: "1.0"},
            "n-shot": {
                self.task_name: 0
            },  # Assuming 0-shot for transformers evaluation
            "higher_is_better": {
                self.task_name: {metric_name: True for metric_name in metrics.keys()}
            },
            "n-samples": {
                self.task_name: {"original": total_samples, "effective": total_samples}
            },
        }

        # Add execution configuration
        results_dict["config"] = {
            "model": self.model_name,
            "model_args": model_args or {},
            "limit": None,
            "bootstrap_iters": 100000,
            "gen_kwargs": {},
        }

        # Prepare samples in lm-eval format
        samples_dict = {self.task_name: self.samples}

        # Use lm-eval's EvaluationTracker to save
        try:
            self.evaluation_tracker.save_results_aggregated(
                results=results_dict, samples=samples_dict
            )
            self.evaluation_tracker.save_results_samples(
                task_name=self.task_name, samples=self.samples
            )
            print(f"Results saved using lm-eval format to: {self.output_path}")

        except Exception as e:
            print(f"Warning: Could not save using lm-eval format: {e}")
            print("Falling back to simple format...")
            self._save_simple_format(metrics)

    def _save_simple_format(self, metrics: Dict[str, float]):
        """Save results in simple format (fallback)."""
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = output_dir / "results_detailed.json"
        with open(results_file, "w") as f:
            json.dump(self.samples, f, indent=2, default=str)

        # Save summary metrics
        if self.samples:
            total = len(self.samples)
            correct = sum(1 for s in self.samples if s.get("is_correct", False))
            accuracy = correct / total if total > 0 else 0.0

            summary = {
                "metrics": metrics,
                "total_samples": total,
                "correct_predictions": correct,
                "accuracy": accuracy,
                "model_name": self.model_name,
                "task_name": self.task_name,
                "timestamp": datetime.now().isoformat(),
            }

            summary_file = output_dir / "results_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

        print(f"Results saved to: {output_dir}")


def create_metric_calculator(task_config) -> MetricCalculator:
    """Create a metric calculator from task configuration using lm-eval metrics."""
    metric_configs = getattr(task_config, "metric_list", [])
    if not metric_configs:
        # Default to exact match if no metrics specified
        metric_configs = [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
        ]

    return MetricCalculator(metric_configs)
