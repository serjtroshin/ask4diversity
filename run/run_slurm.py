import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential SLURM job submitter for src."
    )
    parser.add_argument(
        "--script",
        type=str,
        default="run_generation",
        help="Script to run (run_generation, run_diversity, or run_arithmetic_diversity)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="sequential",
        choices=["sequential", "parallel", "iteration"],
        help="Type of task to run (sequential, parallel, iteration)",
    )
    parser.add_argument(
        "--override_n_iterations",
        type=int,
        default=None,
        help="Override the number of iterations for the task (if applicable)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen4b",
        help="Model to use (e.g., qwen4b, qwen8b)",
    )
    parser.add_argument(
        "--bash",
        action="store_true",
    )
    parser.add_argument(
        "--python_path",
        type=str,
        default="python",
        help="Python interpreter path",
    )
    parser.add_argument("--name", type=str, default="debug", help="Experiment name")
    parser.add_argument(
        "--limit",
        type=str,
        default="null",
        help="Limit parameter (use 'null' if not set)",
    )
    parser.add_argument(
        "--override", action="store_true", help="Override existing outputs"
    )
    parser.add_argument(
        "other_args", nargs=argparse.REMAINDER, help="Other arguments to pass through"
    )
    return parser.parse_args()


def run_sbatch(command: str):
    print(f"Submitting: {command}")
    subprocess.run(command, shell=True, check=True)


def model_config(model_name: str):
    if model_name == "qwen1.7b":
        return " model=qwen3-1.7b generation.batch_size=32"
    elif model_name == "qwen4b":
        return " model=qwen3-4b generation.batch_size=16"
    elif model_name == "qwen8b":
        return " model=qwen3-8b generation.batch_size=8"
    elif model_name == "qwen14b":
        return " model=qwen3-14b generation.batch_size=4"
    elif model_name == "qwen32b":
        return " model=qwen3-32b generation.batch_size=2"
    elif model_name == "ds8b":
        return " model=deepseek-8b generation.batch_size=8"
    elif model_name == "qwen3-4b-instruct":
        return " model=qwen3-4b-instruct generation.batch_size=16"
    elif model_name == "qwen3-4b-thinking":
        return " model=qwen3-4b-thinking generation.batch_size=16"
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_sequential(sbatch: str, base_cmd: str, n_seeds: int = 3):
    # 1. Sequential task with multiple seeds
    for seed in range(0, n_seeds):
        sequential_cmd = f'{sbatch} "{base_cmd} task=zeroshot_gsm8k_sequential generation.seed={seed}"'
        run_sbatch(sequential_cmd)


def run_parallel(
    sbatch: str, base_cmd: str, n_seeds: int = 3, override_n_iterations: int = None
):
    if override_n_iterations is not None:
        optional = f"diversity.calculate_over_first_k={override_n_iterations}"
    else:
        optional = ""
    # 2. Parallel seeds (submitted sequentially)
    for seed in range(0, n_seeds):
        parallel_cmd = f'{sbatch} "{base_cmd} task=zeroshot_gsm8k_parallel generation.seed={seed} {optional}"'
        run_sbatch(parallel_cmd)


def run_iterations(
    sbatch: str, base_cmd: str, n_iterations: int = 3, override_n_iterations: int = None
):
    if override_n_iterations is not None:
        optional = f"diversity.calculate_over_first_k={override_n_iterations}"
    else:
        optional = ""
    # 3. Iteration task
    iteration_cmd = f'{sbatch} "{base_cmd} task=zeroshot_gsm8k_iteration task.nsolutions={n_iterations} {optional}"'
    run_sbatch(iteration_cmd)


def main():
    """
    Example: python run/run_slurm.py --limit 32 --bash model=qwen3-4b
    This script submits SLURM jobs for different sampling methods. OR submits via bash if --bash is set.
    # Metrics:
    pass    --script run_diversity
    """
    args = parse_args()
    if args.bash:
        args.sbatch_job = "bash scripts/submit_task.sh"
    else:
        if args.script == "run_diversity" or args.script == "run_arithmetic_diversity":
            args.sbatch_job = "sbatch scripts/submit_task_cpu.sh"
            print(f"Using CPU job script for {args.script}")
        elif args.script == "run_generation":
            args.sbatch_job = "sbatch scripts/submit_task.sh"
            print("Using GPU job script for run_generation")
        else:
            raise ValueError(f"Unknown script: {args.script}")

    base_cmd = (
        f"PYTHONPATH='.' {args.python_path} src/{args.script}.py "
        f"--config-name run limit={args.limit} override={args.override} "
        f"experiment.name={args.name} {' '.join(args.other_args)}"
    )

    sbatch = args.sbatch_job

    # define model here
    model = args.model  # qwen4b, qwen8b, etc.
    model_args = model_config(model)

    # Add model args to the base command
    base_cmd += model_args

    print(f"Base command: {base_cmd}")

    # Run the tasks
    if args.type == "sequential":
        if args.override_n_iterations is not None:
            raise ValueError(
                "Override n_iterations is not applicable for sequential tasks."
            )
        run_sequential(sbatch, base_cmd, n_seeds=1)
    elif args.type == "parallel":
        n_seeds = 5
        run_parallel(
            sbatch,
            base_cmd,
            n_seeds=n_seeds if args.script == "run_generation" else 1,
            override_n_iterations=args.override_n_iterations,  # if need to run metrics over less number of samples
        )
    elif args.type == "iteration":
        n_iterations = 5
        run_iterations(
            sbatch,
            base_cmd,
            n_iterations=n_iterations,
            override_n_iterations=args.override_n_iterations,
        )
    else:
        raise ValueError(f"Unknown type: {args.type}")

    print("All SLURM jobs submitted.")


if __name__ == "__main__":
    main()
