import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .lmeval_stopping import create_lmeval_stopping_handler

logger = logging.getLogger(__name__)


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers providing a unified interface."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.dtype = self._get_dtype(dtype)
        # Placeholder stopping handler
        self.stopping_handler = create_lmeval_stopping_handler(None)

    @staticmethod
    def _get_dtype(dtype_str: str) -> torch.dtype:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        generation_config: Dict[str, Any],
        batch_size: int = 1,
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def apply_chat_template(
        self, messages: List[Dict[str, str]], enable_thinking: bool = True
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def __del__(self):
        try:
            if hasattr(self, "engine"):
                self.engine.shutdown()
        except Exception:
            pass


class vLLMModelWrapper(BaseModelWrapper):
    """Wrapper for vLLM models using the unified BaseModelWrapper interface via the LLM class."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        max_model_len: Optional[int] = None,
    ):
        super().__init__(model_name, dtype)
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(
            f"Loading vLLM model {model_name} on {self.device} with dtype {dtype}"
        )

        # Initialize LLM engine
        llm_kwargs: Dict[str, Any] = {
            "model": model_name,
            "tokenizer": model_name,
            "tokenizer_mode": "auto",
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": self.dtype,
            "seed": seed or 0,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        self.engine = LLM(**llm_kwargs)
        self.sampling_params = self.engine.get_default_sampling_params()
        logger.info(f"Default sampling params: {self.sampling_params}")

        # Load tokenizer for chat templates
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.until = create_lmeval_stopping_handler(
            self.tokenizer
        ).until  # use stop list
        if isinstance(self.until, str):
            self.until = [self.until]
        if self.until is None:
            self.until = []
        self.sampling_params.until = self.until
        logger.info(f"stopping criteria list: {self.until}")

    def generate(
        self,
        prompts: List[str],
        generation_config: Dict[str, Any],
        batch_size: int = 1,
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
    ) -> List[str]:

        sampling_params: SamplingParams = self.sampling_params
        sampling_params.seed = seed
        sampling_params.max_tokens = max_new_tokens
        sampling_params.stop = self.until + generation_config.get(
            "until", []
        )  # stop strings
        sampling_params.include_stop_str_in_output = (
            True  # since we add </Solution> to stop tokens
        )
        logger.info(f"Used sampling params: {sampling_params}")
        outputs: List[str] = []

        # vLLM generate yields Response objects with 'text' attribute
        for out in self.engine.generate(prompts, sampling_params):
            text = out.outputs[0].text if out.outputs else ""
            outputs.append(text)
        print(f"Generated {len(outputs)} outputs")
        if verbose:
            print(f"First output: {outputs[0] if outputs else 'None'}")
        return outputs

    def apply_chat_template(
        self, messages: List[Dict[str, str]], enable_thinking: bool = True
    ) -> str:
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template
        ):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            return "".join(f"{m.get('role')}: {m.get('content')}\n" for m in messages)

    def count_tokens(self, text: str) -> int:
        try:
            return len(self.engine.tokenizer.encode(text))
        except Exception:
            return len(text.split())
