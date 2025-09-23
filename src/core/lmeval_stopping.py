"""
LM-Eval stopping implementation adapted for our transformers runner.
Imports the actual stopping criteria from lm-evaluation-harness to ensure compatibility.
"""

from typing import List, Union


try:
    # Direct imports from lm-evaluation-harness
    import transformers
    from lm_eval.models.utils import (MultiTokenEOSCriteria,
                                      handle_stop_sequences,
                                      stop_sequences_criteria)
except ImportError as e:
    raise ImportError(f"Could not import lm-eval stopping utilities: {e}")


class LMEvalStoppingHandler:
    """Handler for stopping criteria using lm-eval's implementation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.until = None

    def prepare_stopping_criteria(
        self, until: Union[str, List[str], None], batch_size: int, input_length: int
    ) -> List[str]:
        """Prepare stopping criteria using lm-eval's handle_stop_sequences."""

        self.until = until

        # Get EOS token
        eos_token = None
        if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token:
            eos_token = self.tokenizer.eos_token
        elif (
            hasattr(self.tokenizer, "eos_token_id")
            and self.tokenizer.eos_token_id is not None
        ):
            try:
                eos_token = self.tokenizer.decode(self.tokenizer.eos_token_id)
            except:
                raise ValueError("No EOS token found in tokenizer")

        # Use lm-eval's function to handle stop sequences
        processed_until = handle_stop_sequences(until, eos_token)

        return processed_until

    def create_transformers_stopping_criteria(
        self, stop_sequences: List[str], batch_size: int, input_length: int
    ):
        """Create transformers stopping criteria using lm-eval's implementation."""

        try:

            # Use lm-eval's stopping criteria creation
            criteria_list = stop_sequences_criteria(
                tokenizer=self.tokenizer,
                stop_sequences=stop_sequences,
                initial_decoder_input_length=input_length,
                batch_size=batch_size,
            )

            return criteria_list

        except Exception as e:
            raise RuntimeError(f"Failed to create stopping criteria: {e}")

    def truncate_at_stop_sequences(self, text: str, stop_sequences: List[str]) -> str:
        """Truncate text at the first occurrence of any stop sequence."""
        if not stop_sequences:
            return text

        min_pos = len(text)
        for stop_seq in stop_sequences:
            if stop_seq:  # Skip empty strings
                pos = text.find(stop_seq)
                if pos != -1 and pos < min_pos:
                    min_pos = pos

        return text[:min_pos] if min_pos < len(text) else text


def create_lmeval_stopping_handler(tokenizer) -> LMEvalStoppingHandler:
    """Create an LMEval stopping handler."""
    return LMEvalStoppingHandler(tokenizer)
