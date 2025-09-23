import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          set_seed)

from .lmeval_stopping import create_lmeval_stopping_handler
from .vllm_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)


class TransformersModelWrapper(BaseModelWrapper):
    """Wrapper for HuggingFace Transformers models implementing BaseModelWrapper."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        super().__init__(model_name, dtype)
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info(
            f"Loading Transformers model {model_name} on {self.device} with dtype {dtype}"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=trust_remote_code,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

        # Override stopping handler with tokenizer-based handler
        self.stopping_handler = create_lmeval_stopping_handler(self.tokenizer)

        # Determine max length
        self.max_length = self._get_max_length()

    def _get_max_length(self) -> int:
        attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        ml = getattr(self.tokenizer, "model_max_length", None)
        if ml and ml < 1e18:
            return ml
        return 2048

    def generate(
        self,
        prompts: List[str],
        generation_config: Dict[str, Any],
        batch_size: int = 1,
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        if seed is not None:
            set_seed(seed)

        outputs: List[str] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            outputs.extend(
                self._generate_batch(batch, generation_config, max_new_tokens)
            )
        return outputs

    def _generate_batch(
        self,
        prompts: List[str],
        generation_config: Dict[str, Any],
        max_new_tokens: Optional[int],
    ) -> List[str]:
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(
            self.device
        )

        # Setup stopping
        stopping_criteria = None
        until = generation_config.get("until")
        if until:
            proc = self.stopping_handler.prepare_stopping_criteria(
                until, batch_size=len(prompts), input_length=inputs.input_ids.shape[1]
            )
            stopping_criteria = (
                self.stopping_handler.create_transformers_stopping_criteria(
                    proc,
                    batch_size=len(prompts),
                    input_length=inputs.input_ids.shape[1],
                )
            )

        gen_args = {k: v for k, v in generation_config.items() if k != "until"}
        gen_kwargs = {
            **inputs,
            **gen_args,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_length": self.max_length,
            "max_new_tokens": max_new_tokens,
        }
        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria

        with torch.no_grad():
            out = self.model.generate(**gen_kwargs)
        # Decode
        in_len = inputs.input_ids.shape[1]
        seqs = out[:, in_len:]
        texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        # Post-truncate
        if until:
            proc = self.stopping_handler.prepare_stopping_criteria(
                until, batch_size=len(prompts), input_length=in_len
            )
            texts = [
                self.stopping_handler.truncate_at_stop_sequences(t, proc) for t in texts
            ]
        return texts

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
            # fallback
            return "".join(
                f"{m.get('role','')}: {m.get('content','')}\n" for m in messages
            )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def __del__(self):
        try:
            if hasattr(self, "model"):
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
