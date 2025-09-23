import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"  # Example model name, adjust as needed

# load the model and tokenizer
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.eval()  # Set the model to evaluation mode

cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
# load from cache
model = AutoModelForCausalLM.from_pretrained(
    model_name, cache_dir=cache_dir, torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
