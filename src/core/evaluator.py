import logging
import warnings
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .dataset_loader import DatasetLoader
from .fewshot import FewShotFormatter
from .lmeval_filters import create_lmeval_filter_adapter
from .metrics import LMEvalResultsLogger, create_metric_calculator
from .model_wrapper import BaseModelWrapper, createModelWrapper
from .yaml_parser import load_task_from_file

logger = logging.getLogger(__name__)


class TaskEvaluator:
    """Main evaluator class that orchestrates the evaluation process."""

    def __init__(
        self,
        model_wrapper: str,
        config_file_path: str,
        model_name: str,
        output_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        use_lmeval_format: bool = True,
        only_for_diversity=False,  # not load the model
        n_iterations: Optional[int] = None,
    ):
        self.config_file_path = config_file_path
        self.model_name = model_name
        self.output_path = output_path
        self.use_lmeval_format = use_lmeval_format
        self.n_iterations = n_iterations

        # Load task configuration
        self.config = load_task_from_file(config_file_path)
        logger.info(f"Loaded task configuration from: {config_file_path}")

        # Initialize components
        self.dataset_loader = DatasetLoader(self.config, config_file_path)

        if not only_for_diversity:
            self.model_wrapper: BaseModelWrapper = createModelWrapper(
                model_wrapper, model_name, device, dtype
            )

        self.fewshot_formatter = FewShotFormatter(self.config)
        self.answer_filter = create_lmeval_filter_adapter(self.config)
        self.metric_calculator = create_metric_calculator(self.config)
        self.local_metric_name = self.metric_calculator.metric_configs[0]["metric"]

        # Initialize enhanced results logger
        self.results_logger = LMEvalResultsLogger(
            output_path=output_path,
            model_name=model_name,
            task_name=self.config.task,
            use_lmeval_format=use_lmeval_format,
        )

        logger.info(
            f"Initialized evaluator for task: {self.config.task}, model: {model_name}"
        )

    def evaluate(
        self,
        limit: Optional[int] = None,
        batch_size: int = 1,
        seed: Optional[int] = None,
        use_chat_template: bool = False,
        enable_thinking: bool = True,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run the complete evaluation."""

        # Prepare generation config
        generation_config = self.config.generation_kwargs.copy()

        # Load dataset
        logger.info("Loading dataset...")
        examples = list(self.dataset_loader.get_examples(limit))
        logger.info(f"Loaded {len(examples)} examples")

        if not examples:
            raise ValueError("No examples found in dataset")

        # Run evaluation
        predictions = []
        targets = []

        logger.info("Starting evaluation...")

        # Process examples in batches
        for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
            batch_examples = examples[i : i + batch_size]
            batch_predictions, batch_targets = self._evaluate_batch(
                batch_examples,
                generation_config,
                batch_size,
                seed,
                use_chat_template,
                enable_thinking=enable_thinking,
                start_idx=i,
                max_new_tokens=max_new_tokens,
            )

            predictions.extend(batch_predictions)
            targets.extend(batch_targets)

        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = self.metric_calculator.calculate_metrics(predictions, targets)

        # Log and save results
        self.results_logger.log_final_metrics(metrics)

        # Prepare model args for saving
        model_args = {
            "model_name": self.model_name,
            "generation_config": generation_config,
            "batch_size": batch_size,
            "seed": seed,
            "use_chat_template": use_chat_template,
            "limit": limit,
        }

        self.results_logger.save_results(metrics, model_args)

        return metrics

    def _evaluate_batch(
        self,
        examples: List[Dict[str, Any]],
        generation_config: Dict[str, Any],
        batch_size: int,
        seed: Optional[int],
        use_chat_template: bool,
        enable_thinking: bool = True,
        start_idx: int = 0,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[List[str], List[str]]:
        """Evaluate a batch of examples."""

        # Prepare prompts
        prompts = []
        targets = []

        for i, example in enumerate(examples):
            # Format prompt with few-shot examples
            if use_chat_template:
                formatted_prompt = self.fewshot_formatter.format_fewshot_prompt(
                    example, use_chat_template=True
                )
                # Apply chat template
                if isinstance(formatted_prompt, list):
                    prompt = self.model_wrapper.apply_chat_template(
                        formatted_prompt, enable_thinking=enable_thinking
                    )
                else:
                    warnings.warn(
                        "Formatted prompt is not a list, chat template not applied"
                    )
                    prompt = formatted_prompt
            else:
                prompt = self.fewshot_formatter.format_fewshot_prompt(
                    example, use_chat_template=False
                )

            if i == 0:
                print(f"Formatted prompt: {prompt}")
            prompts.append(prompt)

            # Get expected answer
            expected_answer = self.fewshot_formatter.get_expected_answer(example)
            targets.append(expected_answer)

        # Generate responses
        generated_texts = self.model_wrapper.generate(
            prompts,
            generation_config,
            batch_size=min(batch_size, len(prompts)),
            seed=seed,
            max_new_tokens=max_new_tokens,
        )

        # Process responses
        predictions = []

        for i, (example, prompt, generated_text, target) in enumerate(
            zip(examples, prompts, generated_texts, targets)
        ):
            # Filter the response to extract the answer using lm-eval filters
            filtered_answer = self.answer_filter.apply_filters(
                [generated_text], [example]
            )
            predictions.append(filtered_answer)

            # Check if correct
            is_correct = self.metric_calculator.calculate_metrics(
                [filtered_answer], [target]
            )[self.local_metric_name]

            # Log this sample with doc_id
            doc_id = start_idx + i
            self.results_logger.log_sample(
                example=example,
                prompt=prompt,
                generated_text=generated_text,
                filtered_answer=filtered_answer,
                expected_answer=target,
                is_correct=is_correct,
                doc_id=doc_id,
            )

            # Print progress for debugging
            if i < 5:  # Show first few examples
                print(f"\nExample {doc_id+1}:")
                print(f"Generated: {generated_text}")
                print(f"Filtered: {filtered_answer}")
                print(f"Expected: {target}")
                print(f"Correct: {is_correct}")

        return predictions, targets


def run_evaluation(
    evaluator: TaskEvaluator,
    limit: Optional[int] = None,
    batch_size: int = 1,
    seed: Optional[int] = None,
    use_chat_template: bool = True,
    enable_thinking: bool = True,
    max_new_tokens: Optional[int] = None,
) -> Dict[str, float]:
    """Convenience function to run evaluation with lm-eval format saving."""

    return evaluator.evaluate(
        limit=limit,
        batch_size=batch_size,
        seed=seed,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
    )
