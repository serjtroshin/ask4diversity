import logging
from typing import Optional


from .transformers_wrapper import TransformersModelWrapper
from .vllm_wrapper import BaseModelWrapper, vLLMModelWrapper

logger = logging.getLogger(__name__)


def createModelWrapper(
    model_wrapper: str,
    model_name: str,
    device: str = "auto",
    dtype: str = "bfloat16",
    max_model_len: Optional[int] = None,
) -> BaseModelWrapper:
    """Factory function to create the appropriate model wrapper."""
    if model_wrapper == "transformers":
        logger.info(
            f"Creating Transformers model wrapper for {model_name} on {device} device"
        )
        return TransformersModelWrapper(model_name, device, dtype)
    elif model_wrapper == "vllm":
        logger.info(f"Creating vLLM model wrapper for {model_name} on {device} device")
        return vLLMModelWrapper(model_name, device, dtype, max_model_len=max_model_len)
    else:
        raise ValueError(f"Unknown model wrapper: {model_wrapper}")
