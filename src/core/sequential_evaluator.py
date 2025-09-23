import logging
import re
import warnings
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .dataset_loader import DatasetLoader
from .fewshot import FewShotFormatter
from .lmeval_filters import create_lmeval_filter_adapter
from .metrics import LMEvalResultsLogger, create_metric_calculator
from .model_wrapper import BaseModelWrapper, createModelWrapper
from .yaml_parser import load_task_from_file
from .evaluator import TaskEvaluator

logger = logging.getLogger(__name__)

FAILED_TO_PARSE = "ERROR: Failed to parse solution"


class SequentialTaskEvaluator(TaskEvaluator):
    """Sequential evaluator class that orchestrates the evaluation process.
    Evaluation is done sequentially, n times per prompt.
    """

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
        n_iterations: int = None,
    ):
        super().__init__(
            model_wrapper=model_wrapper,
            config_file_path=config_file_path,
            model_name=model_name,
            output_path=output_path,
            device=device,
            dtype=dtype,
            use_lmeval_format=use_lmeval_format,
            only_for_diversity=only_for_diversity,
            n_iterations=n_iterations,
        )

    def insert_generated_outputs_prompt(
        self, prompt: str, generated_results: List[str]
    ) -> str:
        def wrap_solution(solution: str, idx: int) -> str:
            """Wrap solution in <Solution i> tags."""
            return f"<Solution {idx}>{solution}</Solution {idx}>"

        if len(generated_results) == 0:
            prompt = prompt.replace("{{solutions}}", "No solutions generated yet.")
        else:
            solution_str = ""
            solution_str += "Available solutions are:\n"
            for i, solution in enumerate(generated_results):
                solution_str += wrap_solution(solution, i + 1) + "\n"
            prompt = prompt.replace(r"{{solutions}}", solution_str.strip())
        return prompt

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
        n_iterations = self.n_iterations  # number of iterations to generate solutions

        # Prepare prompts
        def parse_solution(generated_text: str) -> str:
            """Extract the text for the most last occurence of <New Solution>...</New Solution>."""
            # extract the text between <Reasoning i> and </Reasoning i>
            pattern = f"<New Solution>(.*?)</New Solution>"
            matches = re.findall(pattern, generated_text, re.DOTALL)
            for match in matches:
                # remove any leading/trailing whitespace and newlines
                match = match.strip()
                return match
            if len(matches) == 0:
                # if no match found, add a placeholder
                return FAILED_TO_PARSE

        targets = []
        alreaty_generated_solutions = [[] for _ in range(len(examples))]
        generated_texts = [[] for _ in range(len(examples))]
        filtered_answers = [[] for _ in range(len(examples))]
        for n in tqdm(range(n_iterations), desc="Evaluating iterations"):
            # for each iteration, we ask the model to generate a new solution
            prompts = []

            for example, solutions in zip(examples, alreaty_generated_solutions):
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
                prompt = self.insert_generated_outputs_prompt(prompt, solutions)
                prompts.append(prompt)

                # Get expected answer
                expected_answer = self.fewshot_formatter.get_expected_answer(example)
                if n == 0:
                    # only on the first iteration we add the expected answer
                    # to the targets, so that we can compare it later
                    targets.append(expected_answer)

            # Generate responses
            _generated_texts = self.model_wrapper.generate(
                prompts,
                generation_config,
                batch_size=min(batch_size, len(prompts)),
                seed=seed,
                max_new_tokens=max_new_tokens,
            )
            # print(
            #     f"Generated texts: {_generated_texts}"
            # )  # Debugging line to check generated texts
            # input()
            # populate already generated solutions and generated texts
            for i, (example, prompt, generated_text) in enumerate(
                zip(examples, prompts, _generated_texts)
            ):
                # Filter the response to extract the answer using lm-eval filters
                filtered_answer = self.answer_filter.apply_filters(
                    [generated_text], [example]
                )
                alreaty_generated_solutions[i].append(parse_solution(generated_text))
                generated_texts[i].append(generated_text)
                filtered_answers[i].append(filtered_answer)

                # Log this sample with doc_id
                doc_id = start_idx + i
                self.results_logger.log_sample(
                    example=example,
                    prompt=prompt,
                    generated_text=generated_text,
                    filtered_answer=filtered_answer,
                    expected_answer=targets[i],
                    is_correct=False,  # We don't check correctness here
                    doc_id=doc_id,
                    iteration_id=n,
                )

        # return only first predictions (iteration 0)
        predictions = [filtered_answers[i][0] for i in range(len(examples))]
        # target are the same for all iterations
        return predictions, targets
