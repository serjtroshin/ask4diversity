"""
LM-Eval filtering implementation adapted for our transformers runner.
Imports the actual filter classes from lm-evaluation-harness to ensure compatibility.
"""

from typing import Any, Dict, List, Optional

try:
    # Direct imports from lm-evaluation-harness
    from lm_eval.api.filter import Filter
    from lm_eval.filters.extraction import (MultiChoiceRegexFilter,
                                            RegexFilter, WhitespaceFilter)
    from lm_eval.filters.selection import TakeFirstFilter, TakeKFilter
except ImportError as e:
    raise ImportError(f"Could not import lm-eval filters: {e}")


class LMEvalFilterAdapter:
    """Adapter to use lm-eval filters with our data format."""

    def __init__(self, filter_configs: List[Dict[str, Any]]):
        self.filter_configs = filter_configs
        self.filter_pipelines = self._build_filter_pipelines()

    def _build_filter_pipelines(self) -> Dict[str, List[Filter]]:
        """Build filter pipelines from configurations."""
        pipelines = {}

        for filter_config in self.filter_configs:
            name = filter_config.get("name", "default")
            filter_chain = filter_config.get("filter", [])

            pipeline = []
            for filter_def in filter_chain:
                filter_fn = self._create_filter(filter_def)
                if filter_fn:
                    pipeline.append(filter_fn)

            pipelines[name] = pipeline

        return pipelines

    def _create_filter(self, filter_def: Dict[str, Any]) -> Optional[Filter]:
        """Create a filter instance from definition."""
        function_name = filter_def.get("function", "")
        kwargs = {k: v for k, v in filter_def.items() if k != "function"}

        try:
            if function_name == "regex":
                return RegexFilter(**kwargs)
            elif function_name == "multi_choice_regex":
                return MultiChoiceRegexFilter(**kwargs)
            elif function_name == "take_first":
                return TakeFirstFilter(**kwargs)
            elif function_name == "remove_whitespace":
                return WhitespaceFilter(**kwargs)
            elif function_name == "take_first_k":
                return TakeKFilter(**kwargs)
            else:
                raise ValueError(f"Unknown filter function '{function_name}'")
        except Exception as e:
            raise RuntimeError(f"Failed to create filter '{function_name}': {e}")

    def apply_filters(
        self, responses: List[str], docs: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Apply all filter pipelines to responses."""
        results = {}

        # Convert single responses to list format expected by lm-eval filters
        resps_list = [[resp] for resp in responses]

        for name, pipeline in self.filter_pipelines.items():
            current_resps = resps_list

            # Apply each filter in the pipeline
            for filter_obj in pipeline:
                try:
                    current_resps = filter_obj.apply(current_resps, docs)
                    # Convert back to list format if needed
                    if hasattr(current_resps, "__iter__") and not isinstance(
                        current_resps, list
                    ):
                        current_resps = list(current_resps)
                except Exception as e:
                    raise RuntimeError(
                        f"Filter {filter_obj.__class__.__name__} failed: {e}"
                    )

            # Extract the final result
            if current_resps and len(current_resps) > 0:
                # Handle different return formats
                if isinstance(current_resps[0], list):
                    results[name] = current_resps[0][0] if current_resps[0] else ""
                else:
                    results[name] = str(current_resps[0])
            else:
                results[name] = ""

        # Return first successful filter result, or default to first pipeline
        if results:
            # Try to return the first named filter result
            for name in self.filter_pipelines.keys():
                if name in results and results[name]:
                    return results[name]
            # Fallback to first result
            return list(results.values())[0]

        # Final fallback - return original response
        return responses[0] if responses else ""


def create_lmeval_filter_adapter(task_config) -> LMEvalFilterAdapter:
    """Create an LMEval filter adapter from task configuration."""
    filter_configs = getattr(task_config, "filter_list", [])
    if not filter_configs:
        # Default filter configuration
        filter_configs = [{"name": "default", "filter": [{"function": "take_first"}]}]

    return LMEvalFilterAdapter(filter_configs)
