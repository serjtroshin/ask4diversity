from typing import Any, Dict, List

from .dataset_loader import TemplateProcessor
from .yaml_parser import TaskConfig


class FewShotFormatter:
    """Handles few-shot example formatting and injection."""

    def __init__(self, config: TaskConfig):
        self.config = config
        self.template_processor = TemplateProcessor()

    def format_fewshot_prompt(
        self, test_example: Dict[str, Any], use_chat_template: bool = False
    ) -> str:
        """Create a complete prompt with few-shot examples and the test question."""
        if (
            self.config.num_fewshot == 0
            or (examples := self._get_fewshot_examples()) == []
        ):
            examples = []
            # Zero-shot case
            if use_chat_template:
                return self._format_chat_template_zeroshot(test_example)
            else:
                return self._format_single_example(test_example, is_test=True)

        if use_chat_template:
            return self._format_chat_template(examples, test_example)
        else:
            return self._format_concatenated(examples, test_example)

    def _get_fewshot_examples(self) -> List[Dict[str, Any]]:
        """Get few-shot examples from the configuration."""
        if (
            not self.config.fewshot_config
            or "samples" not in self.config.fewshot_config
        ):
            return []

        samples = self.config.fewshot_config["samples"]
        sampler = self.config.fewshot_config.get("sampler", "first_n")

        if sampler == "first_n":
            # Take the first N examples
            return samples[: self.config.num_fewshot]
        else:
            # Other samplers not implemented yet, just take first N
            return samples[: self.config.num_fewshot]

    def _format_single_example(
        self, example: Dict[str, Any], is_test: bool = False
    ) -> str:
        """Format a single example (question + answer for training, question only for test)."""
        # Start with description if available
        prompt = ""
        if self.config.description:
            prompt += self.config.description

        # Add the question part
        question_text = self.template_processor.process_template(
            self.config.doc_to_text, example
        )
        prompt += question_text

        # Add answer for training examples
        if not is_test and self.config.doc_to_target:
            if "target" in example:
                # Use the target field directly for few-shot examples
                answer = example["target"]
            else:
                # Use doc_to_target template for regular examples
                answer = self.template_processor.process_template(
                    self.config.doc_to_target, example
                )

            if answer:
                prompt += answer
                # Add newline if not present
                if not prompt.endswith("\n"):
                    prompt += "\n"

        return prompt

    def _format_concatenated(
        self, examples: List[Dict[str, Any]], test_example: Dict[str, Any]
    ) -> str:
        """Format examples as concatenated text."""
        prompt = ""

        # Add description once at the beginning if available
        if self.config.description:
            prompt += self.config.description

        # Add few-shot examples
        for example in examples:
            example_text = self._format_single_example(example, is_test=False)
            prompt += example_text
            if not prompt.endswith("\n"):
                prompt += "\n"

        # Add test example (question only)
        test_text = self.template_processor.process_template(
            self.config.doc_to_text, test_example
        )
        prompt += test_text

        return prompt

    def _format_chat_template_zeroshot(
        self, test_example: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Format a single test example as a chat message (no explicit tokens)."""

        messages = []

        # Optional system message
        if self.config.description:
            messages.append(
                {"role": "system", "content": self.config.description.strip()}
            )

        # User message with the test input
        test_text = self.template_processor.process_template(
            self.config.doc_to_text, test_example
        )
        messages.append({"role": "user", "content": test_text.strip()})

        return messages

    def _format_chat_template(
        self, examples: List[Dict[str, Any]], test_example: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Format examples as chat messages."""
        messages = []

        # Add system message if description is available
        if self.config.description:
            messages.append(
                {"role": "system", "content": self.config.description.strip()}
            )

        # Add few-shot examples as alternating user/assistant messages
        for example in examples:
            # User message (question)
            question_text = self.template_processor.process_template(
                self.config.doc_to_text, example
            )
            messages.append({"role": "user", "content": question_text})  # .strip()

            # Assistant message (answer)
            if "target" in example:
                answer = example["target"]
            else:
                answer = self.template_processor.process_template(
                    self.config.doc_to_target, example
                )

            if answer:
                messages.append({"role": "assistant", "content": answer})  # .strip()

        # Add test question as final user message
        test_text = self.template_processor.process_template(
            self.config.doc_to_text, test_example
        )
        messages.append({"role": "user", "content": test_text})  # .strip()

        return messages

    def get_expected_answer(self, example: Dict[str, Any]) -> str:
        """Get the expected answer for an example."""
        if self.config.doc_to_target:
            return self.template_processor.process_template(
                self.config.doc_to_target, example
            )
        elif "target" in example:
            return example["target"]
        elif "answer" in example:
            return str(example["answer"])
        else:
            return ""


class MultipleChoiceFormatter:
    """Handles multiple choice specific formatting."""

    def __init__(self, config: TaskConfig):
        self.config = config

    def get_choice_labels(self) -> List[str]:
        """Get the choice labels (e.g., ['(A)', '(B)', '(C)', '(D)'])."""
        if self.config.doc_to_choice:
            return self.config.doc_to_choice
        else:
            # Default for multiple choice tasks
            return ["(A)", "(B)", "(C)", "(D)"]

    def get_choice_logprobs(self, model_wrapper, prompt: str) -> Dict[str, float]:
        """Get log probabilities for each choice."""
        # This would require implementing logprob calculation
        # For now, we'll use generation-based approach
        raise NotImplementedError("Log probability calculation not implemented yet")

    def format_choices(self, example: Dict[str, Any]) -> str:
        """Format the multiple choice options."""
        choices = self.get_choice_labels()
        choice_text = ""

        for i, choice_label in enumerate(choices):
            choice_key = f"choice{i+1}"  # choice1, choice2, etc.
            if choice_key in example:
                choice_text += f"{choice_label} {example[choice_key]}\n"

        return choice_text
