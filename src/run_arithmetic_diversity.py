#!/usr/bin/env python3
"""
Arithmetic diversity evaluation runner.

This script analyzes the diversity of arithmetic expressions in mathematical reasoning outputs.
It reuses the parsing functions from run_diversity.py to avoid code duplication, then:
1. Parse solutions from model responses (using run_diversity.py functions)
2. Extract arithmetic expressions from each solution using LLM
3. Calculate diversity metrics on the arithmetic expressions

This complements run_diversity.py by focusing on mathematical reasoning diversity rather than textual diversity.

Output Files:
- sample_arithmetic_*.json (arithmetic expressions + diversity metrics)
- arithmetic_diversity_report.csv (metrics summary)

Usage:
    # Basic usage (uses config defaults: seed=0, qwen3-1.7b, zeroshot_gsm8k_sequential):
    python src/run_arithmetic_diversity.py

    # Use different model/task/seed (same as run_generation.py):
    python src/run_arithmetic_diversity.py \
        model=qwen3-4b \
        task=zeroshot_gsm8k_parallel \
        generation.seed=1

    # Override judge model or other parameters:
    python src/run_arithmetic_diversity.py \
        arithmetic.model_name="Qwen/Qwen3-4B" \
        arithmetic.verbose=false

    # Skip inference and only compute metrics from existing arithmetic expressions:
    python src/run_arithmetic_diversity.py \
        arithmetic.skip_inference=true

    # Run alongside text diversity analysis:
    python src/run_diversity.py
    python src/run_arithmetic_diversity.py

Uses the same config as run_diversity.py (run.yaml) with arithmetic-specific parameters configured.
"""

import json
import logging
import sys
import time
import ast
import operator
import multiprocessing as mp
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.stats import sem
from filelock import FileLock

# Add the project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core import choose_task_evaluator, TaskEvaluator
from src.core.model_wrapper import createModelWrapper
from src.core.sequential_evaluator import FAILED_TO_PARSE
from src.core.diversity_utils import (
    _distinct_ngram_diversity,
    distinct_ngram_diversity,
    effective_number_of_samples,
)
from src.utils.parse_utils import parse_answer

# Import functions from run_diversity.py to avoid duplication
from src.run_diversity import (
    setup_logging,
    parse_responses_sequential,
    parse_responses_parallel,
    parse_responses_iteration
)

# Register eval resolver if not already registered
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def from_samples_path_arithmetic(sample_file: Path, n_solutions: int, max_samples: int = None) -> Path:
    """Create output path for arithmetic expressions (different from diversity output)"""
    parent_dir = sample_file.parent
    
    # Check if parent_dir already contains n_solutions directory to avoid duplication
    if f"n_solutions={n_solutions}" in str(parent_dir):
        # If n_solutions directory already exists, use the parent directly
        output_dir = parent_dir
    else:
        # Otherwise, add the n_solutions directory
        output_dir = parent_dir / f"n_solutions={n_solutions}"
    
    sample_file_name = sample_file.name
    sample_file_name_with_arithmetic = sample_file_name.replace(
        "sample", "sample_arithmetic"
    ).replace(".jsonl", ".json")
    
    # Add subset indicator to filename if max_samples is specified
    if max_samples is not None:
        sample_file_name_with_arithmetic = sample_file_name_with_arithmetic.replace(
            ".json", f"_subset_{max_samples}.json"
        )
    
    output_file = output_dir / sample_file_name_with_arithmetic
    
    # make directory if it does not exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file


ARITHMETIC_EXTRACTION_PROMPT = """You will receive a math question and a free-form solution. Extract the sequence of arithmetic steps from the solution and output them one by one.

Rules:
- Output ONLY lines made of digits 0-9, parentheses (), the operators + - * / ^, and optionally "=" to show each step's result.
- No words, units, currency symbols, or extra text.
- One step per line, in the order implied by the solution.
- Convert verbal quantities to numbers. Replace references like "the remainder" with the actual numeric value.
- Keep only the steps that lead to the final answer.
- If no computable arithmetic appears, output an empty line.

Example:

Question: Janet lays 16 eggs a day. She eats 3, uses 4 for baking, and sells the rest for $2 each. How much money does she make?
Solution: Janet lays 16 eggs per day. She eats 3 and uses 4 for baking, so 16 - 7 = 9 eggs left. She sells them at $2 each → 9 * 2 = $18.
Output:
3 + 4 = 7
16 - 7 = 9
9 * 2 = 18

Now, extract the arithmetic steps from the following:

Question: {question}
Solution: {solution}
Output:
"""


def add_arithmetic_extraction(
    output_json: Dict[str, Any],
    examples: List[Dict[str, Any]],
    model_wrapper,
    extraction_prompt: str,
    batch_size: int,
    logger: logging.Logger,
    n_solutions: int,
    verbose: bool = False,
    max_samples: int = None
) -> Dict[str, Any]:
    """Add arithmetic extraction to parsed responses from run_diversity.py"""
    # Change output file path to use arithmetic naming
    original_output_file = Path(output_json["output_file"])
    # Extract the sample file name from the original diversity output path
    sample_file_name = original_output_file.name.replace(
        "sample_diversity", "samples"
    ).replace(".json", ".jsonl")
    sample_file_path = original_output_file.parent / sample_file_name
    arithmetic_output_file = from_samples_path_arithmetic(sample_file_path, n_solutions)

    # Extract arithmetic expressions using LLM
    arithmetic_expressions = extract_arithmetic_expressions_batch(
        output_json["parsed_responses"],
        examples,
        model_wrapper,
        extraction_prompt,
        batch_size,
        logger,
        verbose,
    )

    # Create new output with arithmetic data
    arithmetic_output = {
        **output_json,
        "arithmetic_expressions": arithmetic_expressions,
        "output_file": str(arithmetic_output_file),
        "n_solutions": n_solutions,
    }

    return arithmetic_output


def safe_eval_arithmetic(expression: str) -> float:
    """
    Safely evaluate arithmetic expressions containing only numbers and basic operators.
    Returns None if evaluation fails or contains disallowed operations.
    """
    # Define allowed operators
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_operators:
                raise ValueError(f"Operator {type(node.op)} not allowed")
            return allowed_operators[type(node.op)](
                eval_node(node.left), eval_node(node.right)
            )
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_operators:
                raise ValueError(f"Operator {type(node.op)} not allowed")
            return allowed_operators[type(node.op)](eval_node(node.operand))
        else:
            raise ValueError(f"Node type {type(node)} not allowed")

    try:
        # Clean the expression - remove common non-arithmetic parts
        clean_expr = expression.strip()

        # Remove common prefixes/suffixes that might be in the output
        if clean_expr.startswith("Output:"):
            clean_expr = clean_expr[7:].strip()

        # Extract just the arithmetic part (before any equals sign)
        if "=" in clean_expr:
            clean_expr = clean_expr.split("=")[0].strip()

        # Parse and evaluate
        tree = ast.parse(clean_expr, mode="eval")
        result = eval_node(tree.body)
        return float(result)
    except Exception as e:
        return None


def validate_arithmetic_expression(
    expression: str, expected_answer: str
) -> Dict[str, Any]:
    """
    Validate if the arithmetic expression evaluates to the expected answer.
    Returns a dictionary with validation results.
    """
    if not expression or not expected_answer:
        return {
            "is_valid": False,
            "expression_result": None,
            "expected_result": expected_answer,
            "validation_error": "Missing expression or expected answer",
        }

    # If expected_answer is already a parsed number (from parse_answer), use it directly
    if isinstance(expected_answer, str):
        try:
            expected_result = float(expected_answer)
        except ValueError:
            # If it's not a number, try to parse it using parse_answer
            parsed_answer = parse_answer(expected_answer)
            if not parsed_answer:
                return {
                    "is_valid": False,
                    "expression_result": None,
                    "expected_result": expected_answer,
                    "validation_error": "Could not parse expected answer",
                }
            try:
                expected_result = float(parsed_answer)
            except ValueError:
                return {
                    "is_valid": False,
                    "expression_result": None,
                    "expected_result": expected_answer,
                    "validation_error": "Expected answer is not a valid number",
                }
    else:
        # If expected_answer is already a number
        expected_result = float(expected_answer)

    # Evaluate the arithmetic expression
    expression_result = safe_eval_arithmetic(expression)

    if expression_result is None:
        return {
            "is_valid": False,
            "expression_result": None,
            "expected_result": expected_result,
            "validation_error": "Could not evaluate arithmetic expression",
        }

    # Check if results match (with small tolerance for floating point)
    is_valid = abs(expression_result - expected_result) < 1e-10

    return {
        "is_valid": is_valid,
        "expression_result": expression_result,
        "expected_result": expected_result,
        "validation_error": None,
    }


def extract_arithmetic_expressions_batch(
    parsed_responses: List[List[str]],
    examples: List[Dict[str, Any]],
    model_wrapper,
    extraction_prompt: str,
    batch_size: int = 8,
    logger: logging.Logger = None,
    verbose: bool = False,
) -> List[List[str]]:
    """Extract arithmetic expressions from parsed solutions using batched LLM inference."""
    if logger is None:
        logger = logging.getLogger(__name__)

    arithmetic_expressions = []
    start_time = time.time()

    # Flatten all solutions for batch processing
    all_prompts = []
    solution_metadata = (
        []
    )  # Track which solution belongs to which example and solution index

    logger.info("Pre-processing solutions for arithmetic extraction...")
    for example_idx, solutions in enumerate(parsed_responses):
        example_expressions = []
        for solution_idx, solution in enumerate(solutions):
            if solution == FAILED_TO_PARSE:
                example_expressions.append(FAILED_TO_PARSE)
                continue

            # Create prompt for arithmetic extraction
            question = examples[example_idx]["doc"]["question"]
            prompt = extraction_prompt.format(question=question, solution=solution)
            formatted_prompt = model_wrapper.apply_chat_template(
                [{"role": "user", "content": prompt}], enable_thinking=False
            )

            all_prompts.append(formatted_prompt)
            solution_metadata.append((example_idx, solution_idx))
            example_expressions.append(None)  # Placeholder

        arithmetic_expressions.append(example_expressions)

    if not all_prompts:
        logger.warning("No valid solutions found for arithmetic extraction")
        return arithmetic_expressions

    logger.info(
        f"Processing {len(all_prompts)} solutions in batches of size {batch_size}"
    )

    # Process in batches
    all_outputs = []
    total_batches = (len(all_prompts) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(all_prompts), batch_size):
        batch_end = min(batch_idx + batch_size, len(all_prompts))
        batch_prompts = all_prompts[batch_idx:batch_end]

        current_batch = batch_idx // batch_size + 1
        logger.info(
            f"Processing batch {current_batch}/{total_batches} ({len(batch_prompts)} solutions)"
        )

        try:
            # Generate responses for the entire batch
            batch_outputs = model_wrapper.generate(
                prompts=batch_prompts,
                generation_config={},
                batch_size=len(batch_prompts),
                max_new_tokens=2048,
            )
            all_outputs.extend(batch_outputs)

        except Exception as e:
            logger.error(f"Error processing batch {current_batch}: {e}")
            # Add empty strings for failed batch
            all_outputs.extend([""] * len(batch_prompts))

        if verbose and batch_idx == 0:
            print(f"First prompt: {batch_prompts[0]}")
            print(f"First output: {batch_outputs[0]}")
    
    # Assign outputs back to their corresponding positions
    for output_idx, (example_idx, solution_idx) in enumerate(solution_metadata):
        if output_idx < len(all_outputs):
            arithmetic_expression = (
                all_outputs[output_idx].strip() if all_outputs[output_idx] else ""
            )
            arithmetic_expressions[example_idx][solution_idx] = arithmetic_expression
        else:
            arithmetic_expressions[example_idx][solution_idx] = ""
    
    total_time = time.time() - start_time
    logger.info(f"Arithmetic extraction completed in {total_time:.2f}s")

    return arithmetic_expressions


def parse_responses_with_arithmetic(
    cfg: DictConfig,
    output_path: Path,
    n_solutions: int,
    calculate_over_first_k: int,
    logger: logging.Logger,
    evaluator: TaskEvaluator,
    model_wrapper,
    extraction_prompt: str,
    batch_size: int,
    verbose: bool = False,
    max_samples: int = None,  # Limit processing to first N samples (None = all samples)
    skip_inference: bool = False,  # If true, skip inference and only load existing arithmetic expressions
) -> Dict[str, Any]:
    """Parse responses using run_diversity.py functions and add arithmetic extraction."""
    
    # Check if arithmetic output already exists
    sample_files = [
        sample_file for sample_file in output_path.rglob("**/samples_*.jsonl")
    ]
    if not sample_files:
        logger.error(
            f"No sample files found in the output path {output_path}. Exiting."
        )
        return None

    sample_file = sample_files[0]
    arithmetic_output_file = from_samples_path_arithmetic(sample_file, n_solutions, max_samples)
    
    if arithmetic_output_file.exists() and not cfg.override:
        if skip_inference:
            logger.info(
                f"Arithmetic output file {arithmetic_output_file} already exists. Will load existing data for metrics computation."
            )
        else:
            logger.info(
                f"Arithmetic output file {arithmetic_output_file} already exists. Skipping."
            )
            return None

    if calculate_over_first_k != n_solutions:
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
        evaluator=evaluator
    )

    if output_json is None:
        logger.warning(
            f"No valid responses found in the output path {output_path}. Exiting."
        )
        return None

    # Load examples for arithmetic extraction
    sample_files = [
        sample_file for sample_file in output_path.rglob("**/samples_*.jsonl")
    ]
    sample_file = sample_files[0]
    
    # Load only the subset of samples we need for efficiency
    if max_samples is not None:
        # Read only the first max_samples lines for efficiency
        examples = []
        with open(sample_file, "r") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                if line.strip():
                    examples.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(examples)} samples (limited to first {max_samples})")
        
        # Also subset the parsed responses to match
        if "parsed_responses" in output_json:
            output_json["parsed_responses"] = output_json["parsed_responses"][:max_samples]
        if "predictions" in output_json:
            output_json["predictions"] = output_json["predictions"][:max_samples]
        if "targets" in output_json:
            output_json["targets"] = output_json["targets"][:max_samples]
        if "is_corrects" in output_json:
            output_json["is_corrects"] = output_json["is_corrects"][:max_samples]
    else:
        # Load all samples
        with open(sample_file, "r") as f:
            examples = [json.loads(line.strip()) for line in f if line.strip()]
        logger.info(f"Loaded all {len(examples)} samples")
    
    # Add arithmetic extraction to the parsed results (or load existing if skipping inference)
    if skip_inference:
        # Try to load existing arithmetic expressions from file
        arithmetic_output = load_existing_arithmetic_expressions(
            output_json, examples, logger, n_solutions, max_samples, output_path
        )
        if arithmetic_output is None:
            logger.error("Could not find existing arithmetic expressions file. Cannot skip inference.")
            return None
    else:
        arithmetic_output = add_arithmetic_extraction(
            output_json, examples, model_wrapper, extraction_prompt, batch_size, logger, n_solutions, verbose, max_samples
        )

    return arithmetic_output


def load_existing_arithmetic_expressions(
    output_json: Dict[str, Any],
    examples: List[Dict[str, Any]],
    logger: logging.Logger,
    n_solutions: int,
    max_samples: int = None,
    output_path: Path = None
) -> Dict[str, Any]:
    """Load existing arithmetic expressions from file instead of running inference."""
    # Find the arithmetic output file path
    if output_path is not None:
        # Use the output_path passed from the main function
        sample_files = [
            sample_file for sample_file in output_path.rglob("**/samples_*.jsonl")
        ]
    else:
        # Fallback to the old method
        sample_files = [
            sample_file for sample_file in Path(output_json["output_file"]).parent.rglob("**/samples_*.jsonl")
        ]
    
    if not sample_files:
        logger.error("No sample files found to determine arithmetic output path")
        return None
    
    sample_file = sample_files[0]
    arithmetic_output_file = from_samples_path_arithmetic(sample_file, n_solutions, max_samples)
    
    if not arithmetic_output_file.exists():
        logger.error(f"Arithmetic output file {arithmetic_output_file} does not exist. Cannot skip inference.")
        return None
    
    logger.info(f"Loading existing arithmetic expressions from {arithmetic_output_file}")
    
    try:
        with open(arithmetic_output_file, "r") as f:
            existing_data = json.load(f)
        
        # Verify the file contains arithmetic expressions
        if "arithmetic_expressions" not in existing_data:
            logger.error("Existing file does not contain arithmetic_expressions field")
            return None
        
        # Update the output file path to match the current run
        existing_data["output_file"] = str(arithmetic_output_file)
        existing_data["n_solutions"] = n_solutions
        
        logger.info(f"Successfully loaded {len(existing_data['arithmetic_expressions'])} samples with arithmetic expressions")
        return existing_data
        
    except Exception as e:
        logger.error(f"Failed to load existing arithmetic expressions: {e}")
        return None


def get_arithmetic_metrics(output_json: dict) -> dict:
    """Extract arithmetic-specific metrics from the output JSON."""
    metrics = {}

    # Calculate failed to parse count for arithmetic expressions
    failed_to_parse_count = sum(
        sum(1 for expr in expressions if expr == FAILED_TO_PARSE)
        for expressions in output_json["arithmetic_expressions"]
    )
    total_expressions = (
        len(output_json["arithmetic_expressions"]) * output_json["n_solutions"]
    )
    avg_failed_to_parse = (
        failed_to_parse_count / total_expressions if total_expressions > 0 else 0
    )
    metrics["avg_failed_to_parse_arithmetic"] = avg_failed_to_parse

    # Calculate average number of prompts for which all arithmetic expressions failed to parse
    avg_failed_to_parse_prompts = (
        sum(
            1
            for expressions in output_json["arithmetic_expressions"]
            if all(expr == FAILED_TO_PARSE for expr in expressions)
        )
        / len(output_json["arithmetic_expressions"])
        if output_json["arithmetic_expressions"]
        else 0
    )
    metrics["avg_failed_to_parse_prompts_arithmetic"] = avg_failed_to_parse_prompts
    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="run")
def main(cfg: DictConfig):
    """Main entry point."""
    # Set up logging
    setup_logging(cfg.logging.verbosity)
    logger = logging.getLogger(__name__)

    # Use the same path construction as run_generation.py (Hydra automatically constructs the path)
    hydra_cfg = HydraConfig.get()
    output_path = Path(hydra_cfg.runtime.output_dir)

    logger.info(f"Hydra output dir: {output_path}")
    logger.info(f"Looking for samples in: {output_path}")

    if not output_path.exists():
        logger.error(f"Output path {output_path} does not exist.")
        logger.error(
            f"Make sure to run generation first or check your config parameters."
        )
        logger.error(
            f"Expected path follows pattern from defaults.yaml: {cfg.output.base_dir}/seed-{cfg.generation.seed}"
        )
        sys.exit(1)

    # Task configuration
    n_solutions = cfg.task.nsolutions  # how many solutions to find
    calculate_over_first_k = cfg.diversity.calculate_over_first_k
    
    # Sample subsetting configuration
    max_samples = getattr(cfg, 'max_samples', None)  # Number of samples to process (None = all)
    if max_samples is not None:
        logger.info(f"Limiting processing to first {max_samples} samples")

    # Create evaluator (needed for reusing run_diversity functions)
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

    # Get arithmetic extraction parameters from config
    arithmetic_model_name = cfg.arithmetic.model_name
    arithmetic_batch_size = cfg.arithmetic.batch_size
    arithmetic_verbose = cfg.arithmetic.verbose
    arithmetic_skip_inference = cfg.arithmetic.skip_inference
    arithmetic_extraction_prompt = ARITHMETIC_EXTRACTION_PROMPT

    # Create model wrapper for arithmetic extraction (only if not skipping inference)
    model_wrapper = None
    if not arithmetic_skip_inference:
        logger.info(
            f"Initializing model wrapper for arithmetic extraction: {arithmetic_model_name}"
        )
        try:
            model_wrapper = createModelWrapper(
                model_wrapper="vllm",
                model_name=arithmetic_model_name,
                device=cfg.model.device,
                dtype=cfg.model.dtype,
                max_model_len=4096,
            )
        except Exception as e:
            logger.error(f"Failed to initialize model wrapper: {e}")
            sys.exit(1)
    else:
        logger.info("Skipping inference - will only compute metrics from existing arithmetic expressions")
        logger.info("Make sure you have run this script before with skip_inference=false to generate the arithmetic expressions file")

    # Parse responses and extract arithmetic expressions
    output_json = parse_responses_with_arithmetic(
        cfg,
        output_path,
        n_solutions,
        calculate_over_first_k,
        logger,
        evaluator,
        model_wrapper,
        arithmetic_extraction_prompt,
        arithmetic_batch_size,
        verbose=arithmetic_verbose,
        max_samples=max_samples,
        skip_inference=arithmetic_skip_inference
    )

    if output_json is None:
        # Check if this is due to file already existing
        sample_files = [
            sample_file for sample_file in output_path.rglob("**/samples_*.jsonl")
        ]
        if sample_files:
            sample_file = sample_files[0]
            arithmetic_output_file = from_samples_path_arithmetic(sample_file, n_solutions, max_samples)
            if arithmetic_output_file.exists() and not cfg.override:
                if arithmetic_skip_inference:
                    logger.info(f"Arithmetic output file {arithmetic_output_file} already exists and override is False. Cannot skip inference without existing file.")
                    return
                else:
                    logger.info(f"Arithmetic output file {arithmetic_output_file} already exists and override is False. Exiting safely.")
                    return
        
        logger.warning(
            f"No valid responses found in the output path {output_path}. Exiting."
        )
        return

    arithmetic_expressions = output_json["arithmetic_expressions"]
    output_file = output_json["output_file"]
    n_solutions = output_json["n_solutions"]

    # Calculate diversity metrics on arithmetic expressions
    logger.info(f"Calculating diversity metrics on {len(arithmetic_expressions)} samples...")
    with mp.Pool(mp.cpu_count()) as pool:
        diversity_scores = pool.starmap(
            distinct_ngram_diversity,
            [
                (arithmetic_expressions[i], cfg.diversity.n, FAILED_TO_PARSE, True)
                for i in range(len(arithmetic_expressions))
            ],
        )

    with mp.Pool(mp.cpu_count()) as pool:
        effective_n_samples = pool.starmap(
            effective_number_of_samples,
            [
                (
                    arithmetic_expressions[i],
                    cfg.diversity.n,
                    FAILED_TO_PARSE
                )
                for i in range(len(arithmetic_expressions))
            ],
        )

    with mp.Pool(mp.cpu_count()) as pool:
        diversity_scores_1gram = pool.starmap(
            distinct_ngram_diversity,
            [
                (arithmetic_expressions[i], 1, FAILED_TO_PARSE, True)
                for i in range(len(arithmetic_expressions))
            ],
        )

    with mp.Pool(mp.cpu_count()) as pool:
        effective_n_samples_1gram = pool.starmap(
            effective_number_of_samples,
            [
                (
                    arithmetic_expressions[i],
                    1,
                    FAILED_TO_PARSE
                )
                for i in range(len(arithmetic_expressions))
            ],
        )

    with mp.Pool(mp.cpu_count()) as pool:
        math_em_steps_diversity_scores = pool.starmap(
            _distinct_ngram_diversity,
            [
                (arithmetic_expressions[i], 1, FAILED_TO_PARSE, True, "math_steps")
                for i in range(len(arithmetic_expressions))
            ],
        )

    with mp.Pool(mp.cpu_count()) as pool:
        math_em_full_diversity_scores = pool.starmap(
            _distinct_ngram_diversity,
            [
                (arithmetic_expressions[i], 1, FAILED_TO_PARSE, True, "math_expressions")
                for i in range(len(arithmetic_expressions))
            ],
        )

    with mp.Pool(mp.cpu_count()) as pool:
        math_em_steps_effective_n_samples = pool.starmap(
            effective_number_of_samples,
            [
                (
                    arithmetic_expressions[i],
                    1,
                    FAILED_TO_PARSE,
                    "math_steps"
                )
                for i in range(len(arithmetic_expressions))
            ],
        )
    
    with mp.Pool(mp.cpu_count()) as pool:
        math_em_full_effective_n_samples = pool.starmap(
            effective_number_of_samples,
            [
                (arithmetic_expressions[i], 1, FAILED_TO_PARSE, "math_expressions")
                for i in range(len(arithmetic_expressions))
            ],
        )
    
    if cfg.arithmetic.verbose:
        print("1st Example:")
        print("initial_response", output_json["parsed_responses"][0])
        print("initial_response_diversity", distinct_ngram_diversity(output_json["parsed_responses"][0], 1, FAILED_TO_PARSE, True, "default"))
        print("initial_response_effective_n_samples", effective_number_of_samples(output_json["parsed_responses"][0], 1, FAILED_TO_PARSE, "default"))
        print("arithmetic_expressions", arithmetic_expressions[0])
        print("diversity_scores", diversity_scores[0])
        print("diversity_scores_1gram", diversity_scores_1gram[0])
        print("effective_n_samples_1gram", effective_n_samples_1gram[0])
        print("math_em_steps_diversity_scores", math_em_steps_diversity_scores[0])
        print("math_em_full_diversity_scores", math_em_full_diversity_scores[0])
        print("math_em_steps_effective_n_samples", math_em_steps_effective_n_samples[0])
        print("math_em_full_effective_n_samples", math_em_full_effective_n_samples[0])

    metrics = {
        "avg_parsed_arithmetic": np.mean(
            [len(expressions) for expressions in arithmetic_expressions]
        ).item(),
        "avg_diversity_arithmetic": np.mean(diversity_scores).item(),
        "std_diversity_arithmetic": sem(diversity_scores).item(),
        "avg_effective_n_samples_arithmetic": np.mean(effective_n_samples).item(),
        "std_effective_n_samples_arithmetic": sem(effective_n_samples).item(),
        "avg_diversity_arithmetic_1gram": np.mean(diversity_scores_1gram).item(),
        "std_diversity_arithmetic_1gram": sem(diversity_scores_1gram).item(),
        "avg_effective_n_samples_arithmetic_1gram": np.mean(effective_n_samples_1gram).item(),
        "std_effective_n_samples_arithmetic_1gram": sem(effective_n_samples_1gram).item(),
        "avg_math_em_steps_diversity": np.mean(math_em_steps_diversity_scores).item(),
        "std_math_em_steps_diversity": sem(math_em_steps_diversity_scores).item(),
        "avg_math_em_full_diversity": np.mean(math_em_full_diversity_scores).item(),
        "std_math_em_full_diversity": sem(math_em_full_diversity_scores).item(),
        "avg_math_em_steps_effective_n_samples": np.mean(math_em_steps_effective_n_samples).item(),
        "std_math_em_steps_effective_n_samples": sem(math_em_steps_effective_n_samples).item(),
        "avg_math_em_full_effective_n_samples": np.mean(math_em_full_effective_n_samples).item(),
        "std_math_em_full_effective_n_samples": sem(math_em_full_effective_n_samples).item(),
    }
    metrics.update(get_arithmetic_metrics(output_json))

    print("Arithmetic Expression Diversity Metrics:", metrics, flush=True)
    output_json["diversity_scores_arithmetic"] = diversity_scores
    output_json["diversity_scores_arithmetic_1gram"] = diversity_scores_1gram
    output_json["effective_n_samples_arithmetic_1gram"] = effective_n_samples_1gram
    output_json["math_em_steps_diversity_scores"] = math_em_steps_diversity_scores
    output_json["math_em_full_diversity_scores"] = math_em_full_diversity_scores
    output_json["math_em_steps_effective_n_samples"] = math_em_steps_effective_n_samples
    output_json["math_em_full_effective_n_samples"] = math_em_full_effective_n_samples

    # Save the results to output file
    with open(output_file, "w") as f:
        json.dump(
            output_json,
            f,
            indent=4,
        )
    logger.info(
        f"Saved arithmetic diversity results to {output_file}. "
        f"Average: {metrics['avg_diversity_arithmetic']}, Std: {metrics['std_diversity_arithmetic']}.\n"
        f"Average 1gram: {metrics['avg_diversity_arithmetic_1gram']}\n"
        f"Math EM Steps Diversity: {metrics['avg_math_em_steps_diversity']:.4f}\n"
        f"Math EM Full Diversity: {metrics['avg_math_em_full_diversity']:.4f}\n"
        f"Average effective n samples: {metrics['avg_effective_n_samples_arithmetic']}, Std: {metrics['std_effective_n_samples_arithmetic']}.\n"
        f"Average effective n samples 1gram: {metrics['avg_effective_n_samples_arithmetic_1gram']}\n"
        f"Math EM Steps Effective N Samples: {metrics['avg_math_em_steps_effective_n_samples']:.4f}\n"
        f"Math EM Full Effective N Samples: {metrics['avg_math_em_full_effective_n_samples']:.4f}\n"
    )

    report_file = cfg.diversity.report_file.replace("diversity_report.csv", "arithmetic_diversity_report.csv")
    lock_file = Path(report_file).with_suffix(".lock")

    with FileLock(lock_file, timeout=10):
        if not Path(report_file).exists():
            with open(report_file, "w") as f:
                f.write("task_type,task_name,model_name,n_solutions,output_file")
                for key in metrics:
                    f.write(f",{key}")
                f.write("\n")

        with open(report_file, "a") as f:
            f.write(
                f"{cfg.task.type},{cfg.task.name},{cfg.model.name},{n_solutions},{output_file}"
            )
            for key in metrics:
                f.write(f",{metrics[key]}")
            f.write("\n")


if __name__ == "__main__":
    main()
