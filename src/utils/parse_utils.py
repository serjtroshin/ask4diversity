import json
import re

FAILED_TO_PARSE = "<Failed to parse>"


def load_samples(samples_path):
    """
    Load samples from a JSONL file.
    """
    samples = []
    with open(samples_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
    # issue: for gsm8k, duplicate samples
    if "gsm8k" in samples_path:
        samples = samples[
            : len(samples) // 2
        ]  # Keep only the first half of the samples (somehow they duplicate)
    return samples


def extract_reasonings(text, n_reasonings=3):
    """
    We parse the content of <Reasoning {reasoning_id}>(.*?)</Reasoning {reasoning_id}>" blocks
    """
    reasoning_steps = []
    # parsing each reasoning
    for reasoning_id in range(1, n_reasonings + 1):
        # extract the text between <Reasoning i> and </Reasoning i>
        pattern = f"<Reasoning {reasoning_id}>(.*?)</Reasoning {reasoning_id}>"
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            # remove any leading/trailing whitespace and newlines
            match = match.strip()
            if match:
                reasoning_steps.append(match)
        if len(matches) == 0:
            # if no match found, add a placeholder
            reasoning_steps.append(FAILED_TO_PARSE)
    return reasoning_steps


def extract_thinking_steps(text):
    """
    We parse the content of <think>...</think> blocks
    """
    # <think> is part of the input, the output starts with thinking
    # we need to extract everything before </think>
    pattern = r"(.*?)</think>"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        return FAILED_TO_PARSE
    return matches[0]


def parse_sample(sample, n_reasonings=3):
    """
    sample json to simpler dict
    n_reasonings: int = 3
        number of reasonings to extract from the sample
    """
    doc_id = sample["doc_id"]
    doc = sample["doc"]
    target = sample["target"]
    resps = sample["resps"][0][0]
    exact_match = sample["exact_match"]
    reasoning_steps = extract_reasonings(resps, n_reasonings=n_reasonings)
    thinking_steps = extract_thinking_steps(resps)
    return {
        "doc_id": doc_id,
        "doc": doc,
        "target": target,
        "resps": resps,
        "exact_match": exact_match,
        "reasoning_steps": reasoning_steps,
        "thinking_steps": thinking_steps,
    }


def parse_answer(answer) -> str | None:
    """
    Parse the answer from a reasoning.
    answer standartization: removes dollar sign and commas, removes last dot.

    Regexp: original filter for an answer from evaluation config:
    - filter:
    - function: regex
        group_select: -1
        regex_pattern: (-?[$0-9.,]{2,})|(-?[0-9]+)
    - function: take_first
    name: flexible-extract
    """
    # 1. extract the answer
    answer = answer.strip()
    regexp_pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    matches = re.findall(regexp_pattern, answer)
    if not matches:
        return None
    # Take the last match, which is the most likely answer
    answer = matches[-1][0] if matches[-1][0] else matches[-1][1]
    # Remove dollar sign and commas, convert to float
    answer = answer.replace("$", "").replace(",", "")
    # remove dots from the end of the answer
    answer = answer.rstrip(".")
    return answer.strip()
