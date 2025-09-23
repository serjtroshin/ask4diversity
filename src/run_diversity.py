#!/usr/bin/env python3
"""
Simple evaluation runner for diversity.
"""

import json
import logging
import re
import sys
from pathlib import Path
import multiprocessing as mp
from typing import List, Tuple

from filelock import FileLock
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.stats import sem

from src.core import choose_task_evaluator, TaskEvaluator

from src.core.sequential_evaluator import FAILED_TO_PARSE
from src.core.diversity_utils import (
    distinct_ngram_diversity,
    effective_number_of_samples,
)

OmegaConf.register_new_resolver("eval", eval)

# Add the project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


def setup_logging(verbosity: str = "INFO"):
    """Set up logging configuration."""
    log_level = getattr(logging, verbosity.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def count_solutions(generated_text: str) -> int:
    """Count the number of solutions in the generated text"""
    # Look for patterns like <Solution 1>, <Solution 2>, etc.
    pattern = r"<Solution (\d+)>"
    matches = re.findall(pattern, generated_text)
    if matches:
        # Convert to integers and find the maximum
        solution_numbers = [int(match) for match in matches]
        return max(solution_numbers)
    else:
        return 0


def parse_solution(
    generated_text: str, solution_id=None, pattern=f"<Solution>(.*?)</Solution>"
) -> str:
    """Extract the text of solutions"""
    # extract the text between <Reasoning i> and </Reasoning i>
    matches = re.findall(pattern, generated_text, re.DOTALL)
    for match in matches:
        # remove any leading/trailing whitespace and newlines
        match = match.strip()
        return match
    if len(matches) == 0:
        # if no match found, add a placeholder
        return FAILED_TO_PARSE


def from_samples_path(sample_file: Path, n_solutions: int) -> Path:
    parent_dir = sample_file.parent
    sample_file_name = sample_file.name
    if sample_file_name.startswith("samples"):
        sample_file_name_with_diversity = sample_file_name.replace(
            "samples", "sample_diversity"
        ).replace(".jsonl", ".json")
    else:
        sample_file_name_with_diversity = sample_file_name.replace(
            "sample", "sample_diversity"
        ).replace(".jsonl", ".json")
    output_file = (
        parent_dir / f"n_solutions={n_solutions}" / sample_file_name_with_diversity
    )
    # make directory if it does not exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file


def parse_responses_sequential(
    output_path: Path,
    n_solutions: int,
    logger: logging.Logger,
    override: bool = False,
    evaluator: TaskEvaluator = None,
    parse_all_solutions: bool = True,
) -> Tuple[List[List[str]], Path | None]:
    """
    sequential samples are located in a single file.
    """
    sample_files = [
        sample_file for sample_file in output_path.rglob("**/samples_*.jsonl")
    ]
    assert (
        len(sample_files) == 1
    ), f"Expected exactly one sample file in the output directory {output_path}."
    sample_file = sample_files[0]
    print(f"Processing sample file: {sample_file}", flush=True)
    output_file = from_samples_path(sample_file, n_solutions)
    print(f"Output file will be: {output_file}", flush=True)

    if output_file.exists():
        if override:
            logger.warning(
                f"Output file {output_file} already exists. Overriding due to --override flag."
            )
        else:
            logger.info(f"Output file {output_file} already exists. Skipping.")
            return [], None
    # load jsonl file
    with open(sample_file, "r") as f:
        examples = [json.loads(line.strip()) for line in f if line.strip()]
        responses = [example["resps"][0][0] for example in examples]
        parsed_responses = [[] for _ in range(len(responses))]
        predictions = [[] for _ in range(len(responses))]
        is_corrects = [[] for _ in range(len(responses))]
        targets = [example["target"] for example in examples]
        for i, response in enumerate(responses):
            if parse_all_solutions:
                n_solutions = count_solutions(response)

            # parse the response to extract n solutions
            for j in range(1, n_solutions + 1):
                parsed_response = parse_solution(
                    response, j, pattern=f"<Solution {j}>(.*?)</Solution {j}>"
                )
                if parsed_response != FAILED_TO_PARSE:
                    parsed_responses[i].append(parsed_response)
                else:
                    parsed_responses[i].append(FAILED_TO_PARSE)

                # Filter the response to extract the answer using lm-eval filters
                filtered_answer = evaluator.answer_filter.apply_filters(
                    [parsed_response], [examples[i]]
                )
                predictions[i].append(filtered_answer)

                # Check if correct
                is_correct = evaluator.metric_calculator.calculate_metrics(
                    [filtered_answer], [targets[i]]
                )[evaluator.local_metric_name]
                is_corrects[i].append(is_correct)

        logger.info(f"Loaded {len(responses)} responses from {sample_file}")
        print(
            "Example parsed response:\n",
            "\n".join(parsed_responses[0]),
        )
    if not responses:
        logger.warning(f"No valid responses found in {sample_file}. Skipping.")
        return None

    return {
        "parsed_responses": parsed_responses,
        "output_file": output_file,
        "predictions": predictions,
        "targets": targets,
        "is_corrects": is_corrects,
    }


def parse_responses_parallel(
    output_path: Path,
    n_solutions: int,
    logger: logging.Logger,
    override: bool = False,
    evaluator: TaskEvaluator = None,
) -> Tuple[List[List[str]], Path | None]:
    """solutions are in multiuple files by seed."""

    sample_files = [
        sample_file for sample_file in output_path.rglob("**/samples_*.jsonl")
    ]
    try:
        sample_file = sample_files[0]
    except IndexError:
        logger.error(
            f"No sample files found in {output_path} Expected files with pattern '**/samples_*.jsonl'."
        )
        exit(1)
    print(f"Processing sample file: {sample_file}", flush=True)
    output_file = from_samples_path(sample_file, n_solutions)

    def extract_seed(path):
        m = re.search(r"/seed-(\d+)/", path)
        return int(m.group(1)) if m else None

    # get any file and parse all files with seeds and the same structure
    seed_orig = extract_seed(str(sample_file))
    assert seed_orig == 0  # let's use only seed 0 for now for "main" folder
    print(f"Extracted seed: {seed_orig}", flush=True)
    if seed_orig is None:
        logger.error(
            f"Seed not found in the sample file name {sample_file.name}. Expected format: */seed-(\d+)/*"
        )
        sys.exit(1)
    with open(sample_file, "r") as f:
        _examples = [json.loads(line.strip()) for line in f if line.strip()]
    _responses = [example["resps"][0][0] for example in _examples]
    parsed_responses = [[] for _ in range(len(_responses))]
    predictions = [[] for _ in range(len(_responses))]
    is_corrects = [[] for _ in range(len(_responses))]
    targets = [example["target"] for example in _examples]

    for seed in range(n_solutions):
        # seed has to be in this range
        sample_file_cur = str(sample_file).replace(f"seed-{seed_orig}", f"seed-{seed}")
        sample_file_cur_parent = Path(sample_file_cur).parent
        # try to find any samples there
        sample_file_cur = next(
            sample_file_cur_parent.rglob(f"samples*.jsonl"),
            None,
        )
        if sample_file_cur is None:
            logger.warning(
                f"No sample file found for seed {seed} in {sample_file_cur_parent}. Skipping."
            )
            continue
        print(f"Processing sample file: {sample_file_cur}", flush=True)
        # assert that the file exists
        if not Path(sample_file_cur).exists():
            logger.error(f"Sample file {sample_file_cur} does not exist.")
            exit(1)

        with open(sample_file_cur, "r") as f:
            examples = [json.loads(line.strip()) for line in f if line.strip()]
            responses = [example["resps"][0][0] for example in examples]
            assert len(responses) == len(
                parsed_responses
            ), f"Expected {len(parsed_responses)} responses, but got {len(responses)}"

            for i, response in enumerate(responses):
                parsed_response = parse_solution(
                    response, None, pattern=f"<Solution>(.*?)</Solution>"
                )
                if parsed_response != FAILED_TO_PARSE:
                    parsed_responses[i].append(parsed_response)
                else:
                    parsed_responses[i].append(FAILED_TO_PARSE)

                # Filter the response to extract the answer using lm-eval filters
                filtered_answer = evaluator.answer_filter.apply_filters(
                    [parsed_response], [examples[i]]
                )
                predictions[i].append(filtered_answer)

                # Check if correct
                is_correct = evaluator.metric_calculator.calculate_metrics(
                    [filtered_answer], [targets[i]]
                )[evaluator.local_metric_name]
                is_corrects[i].append(is_correct)
    print(
        "Example parsed response:\n",
        "\n".join(parsed_responses[0]),
    )
    return {
        "parsed_responses": parsed_responses,
        "output_file": output_file,
        "predictions": predictions,
        "targets": targets,
        "is_corrects": is_corrects,
    }


def parse_responses_iteration(
    output_path: Path,
    n_solutions: int,
    logger: logging.Logger,
    override: bool = False,
    evaluator: TaskEvaluator = None,
) -> Tuple[List[List[str]], Path | None]:
    """For this sampling method, all solutions are in one file."""
    sample_files = [
        sample_file for sample_file in output_path.rglob("**/samples_*.jsonl")
    ]
    assert (
        len(sample_files) == 1
    ), "Expected exactly one sample file in the output directory."
    sample_file = sample_files[0]
    print(f"Processing sample file: {sample_file}", flush=True)
    output_file = from_samples_path(sample_file, n_solutions)
    print(f"Output file will be: {output_file}", flush=True)

    if output_file.exists():
        if override:
            logger.warning(
                f"Output file {output_file} already exists. Overriding due to --override flag."
            )
        else:
            logger.info(f"Output file {output_file} already exists. Skipping.")
            exit(0)

    with open(sample_file, "r") as f:
        examples = [json.loads(line.strip()) for line in f if line.strip()]
        # responses = [example["resps"][0][0] for example in examples]
        max_iteration_id = max(jsn["iteration_id"] for jsn in examples)
        max_doc_id = max(jsn["doc_id"] for jsn in examples)
        assert (
            n_solutions <= max_iteration_id + 1
        ), f"Expected n_solutions ({n_solutions}) to be less than or equal to max iteration id ({max_iteration_id})+1"

        parsed_responses = [
            [None for __ in range(n_solutions)] for _ in range(max_doc_id + 1)
        ]  # now we map
        predictions = [
            [None for __ in range(n_solutions)] for _ in range(len(parsed_responses))
        ]
        is_corrects = [
            [None for __ in range(n_solutions)] for _ in range(len(parsed_responses))
        ]
        targets = [None for _ in range(len(parsed_responses))]

        for example in examples:
            iteration_id = example["iteration_id"]
            if iteration_id >= n_solutions:
                #     logger.warning(
                #         f"Skipping example with iteration_id {iteration_id} as it exceeds n_solutions {n_solutions}."
                #     )
                continue
            doc_id = example["doc_id"]
            generated_text = example["resps"][0][0]
            target = example["target"]
            parsed_response = parse_solution(
                generated_text, None, pattern=f"<New Solution>(.*?)</New Solution>"
            )
            if parsed_response != FAILED_TO_PARSE:
                parsed_responses[doc_id][iteration_id] = parsed_response
            else:
                parsed_responses[doc_id][iteration_id] = FAILED_TO_PARSE

            # Filter the response to extract the answer using lm-eval filters
            filtered_answer = evaluator.answer_filter.apply_filters(
                [parsed_response], [example]
            )
            predictions[doc_id][iteration_id] = filtered_answer

            # Check if correct
            is_correct = evaluator.metric_calculator.calculate_metrics(
                [filtered_answer], [target]
            )[evaluator.local_metric_name]
            is_corrects[doc_id][iteration_id] = is_correct
            targets[doc_id] = target
    print(
        "Example parsed response:\n",
        "\n".join(parsed_responses[0]),
    )
    return {
        "parsed_responses": parsed_responses,
        "output_file": output_file,
        "predictions": predictions,
        "targets": targets,
        "is_corrects": is_corrects,
    }


def parse_responses(
    cfg: DictConfig,
    output_path: Path,
    n_solutions: int,
    calculate_over_first_k: int,
    logger: logging.Logger,
    evaluator: TaskEvaluator,
) -> Tuple[List[List[str]], Path | None]:
    """
    Parse responses based on the task type.
    output_path: Path to the output directory where samples are stored.
    n_solutions: Number of solutions to find.
    calculate_over_first_k: Number of solutions to use for metrics calculation.
    """
    if calculate_over_first_k != n_solutions:
        # assert (
        #    calculate_over_first_k <= n_solutions
        # ), f"calculate_over_first_k ({calculate_over_first_k}) should be less than or equal to n_solutions ({n_solutions})"
        n_solutions = calculate_over_first_k

    logger.info(f"Task type: {cfg.task.type}")
    if cfg.task.type == "sequential":
        parser = parse_responses_sequential
    elif cfg.task.type == "parallel":
        parser = parse_responses_parallel
    elif cfg.task.type == "iteration":
        parser = parse_responses_iteration
    else:
        logger.error(f"Unknown task type: {cfg.task.type}. Exiting.")
        sys.exit(1)
    output_json = parser(
        output_path,
        n_solutions=n_solutions,
        logger=logger,
        override=cfg.override,
        evaluator=evaluator,
    )
    if output_json is None:
        logger.warning(
            f"No valid responses found in the output path {output_path}. Exiting."
        )
        return None
    output_json["output_file"] = str(output_json["output_file"])  # ensure it's a string
    output_json["n_solutions"] = n_solutions
    return output_json


def get_metrics(output_json: dict) -> dict:
    """Extract metrics from the output JSON."""
    metrics = {}
    # calculated the average number of failed to parse solutions
    failed_to_parse_count = sum(
        sum(1 for response in responses if response == FAILED_TO_PARSE)
        for responses in output_json["parsed_responses"]
    )
    total_solutions = len(output_json["parsed_responses"]) * output_json["n_solutions"]
    avg_failed_to_parse = (
        failed_to_parse_count / total_solutions if total_solutions > 0 else 0
    )
    metrics["avg_failed_to_parse"] = avg_failed_to_parse

    # calculate the average number of prompts for which all solutions failed to parse
    avg_failed_to_parse_prompts = (
        sum(
            1
            for responses in output_json["parsed_responses"]
            if all(response == FAILED_TO_PARSE for response in responses)
        )
        / len(output_json["parsed_responses"])
        if output_json["parsed_responses"]
        else 0
    )
    metrics["avg_failed_to_parse_prompts"] = avg_failed_to_parse_prompts
    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="run")
def main(cfg: DictConfig):
    """Main entry point."""
    # Set up logging
    setup_logging(cfg.logging.verbosity)
    logger = logging.getLogger(__name__)
    hydra_cfg = HydraConfig.get()
    output_path = Path(hydra_cfg.runtime.output_dir)
    print(
        f"Hydra output dir: {output_path}. This script will look for samples in this directory.",
        flush=True,
    )
    if not output_path.exists():
        logger.error(f"Output path {output_path} does not exist. Exiting.")
        sys.exit(1)

    # task
    n_solutions = cfg.task.nsolutions  # how many solutions to find
    calculate_over_first_k = cfg.diversity.calculate_over_first_k

    evaluator: TaskEvaluator = choose_task_evaluator(cfg.task.type)(
        model_wrapper=cfg.generation.model_wrapper,
        config_file_path=cfg.task.task_config,
        model_name=cfg.model.name,
        output_path=output_path,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
        use_lmeval_format=True,
        only_for_diversity=True,
    )

    output_json = parse_responses(
        cfg,
        output_path,
        n_solutions,
        calculate_over_first_k,
        logger,
        evaluator,
    )
    if output_json is None:
        logger.warning(
            f"No valid responses found in the output path {output_path}. Exiting."
        )
        return
    parsed_responses = output_json["parsed_responses"]
    output_file = output_json["output_file"]
    n_solutions = output_json["n_solutions"]
    if output_file is None:
        logger.info("No valid responses found. Exiting.")
        return

    # use multiprocessing to calculate diversity
    with mp.Pool(mp.cpu_count()) as pool:
        diversity_scores = pool.starmap(
            distinct_ngram_diversity,
            [
                (parsed_responses[i], cfg.diversity.n, FAILED_TO_PARSE)
                for i in range(len(parsed_responses))
            ],
        )
    with mp.Pool(mp.cpu_count()) as pool:
        effective_n_samples = pool.starmap(
            effective_number_of_samples,
            [
                (
                    parsed_responses[i],
                    cfg.diversity.n,
                    FAILED_TO_PARSE,
                )
                for i in range(len(parsed_responses))
            ],
        )

    metrics = {
        "avg_parsed_solutions": np.mean(
            [len(responses) for responses in parsed_responses]
        ).item(),
        "avg_diversity": np.mean(diversity_scores).item(),
        "std_diversity": sem(diversity_scores).item(),
        "avg_effective_n_samples": np.mean(effective_n_samples).item(),
        "std_effective_n_samples": sem(effective_n_samples).item(),
    }
    metrics.update(get_metrics(output_json))  # other metrics like failed to parse
    print("metrics", metrics, flush=True)
    output_json["diversity_scores"] = diversity_scores

    # save the results to output file
    with open(output_file, "w") as f:
        json.dump(
            output_json,
            f,
            indent=4,
        )
    logger.info(
        f"Saved diversity results to {output_file}. Average: {metrics['avg_diversity']}, Std: {metrics['std_diversity']}"
    )

    # save number to a report file via append mode: save as csv
    report_file = cfg.diversity.report_file
    print(f"Saving results to report file: {report_file}", flush=True)
    lock_file = Path(report_file).with_suffix(
        ".lock"
    )  # e.g. outputs/diversity_report.lock
    # check if file exists, if not create it with headers
    with FileLock(lock_file, timeout=10):  # wait max 10 seconds for lock
        if not Path(report_file).exists():
            with open(report_file, "w") as f:
                f.write("task_type,task_name,model_name,n_solutions,output_file")
                for key in metrics:
                    f.write(f",{key}")
                f.write("\n")
        # append the results
        with open(report_file, "a") as f:
            f.write(
                f"{cfg.task.type},{cfg.task.name},{cfg.model.name},{n_solutions},{output_file}"
            )
            for key in metrics:
                f.write(f",{metrics[key]}")
            f.write("\n")


if __name__ == "__main__":
    main()
