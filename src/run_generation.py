#!/usr/bin/env python3
"""
Transformers-based evaluation runner.
Replaces the lm-evaluation-harness based run.py with direct transformers usage.
"""

import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.core import choose_task_evaluator

OmegaConf.register_new_resolver("eval", eval)

# Add the project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.evaluator import run_evaluation


def setup_logging(verbosity: str = "INFO"):
    """Set up logging configuration."""
    log_level = getattr(logging, verbosity.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@hydra.main(version_base=None, config_path="configs", config_name="run")
def main(cfg: DictConfig):
    """Main entry point."""
    # Set up logging
    setup_logging(cfg.logging.verbosity)
    logger = logging.getLogger(__name__)
    hydra_cfg = HydraConfig.get()
    output_path = Path(hydra_cfg.runtime.output_dir)
    logger.info(f"Hydra output dir: {output_path}")
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info("Starting transformers-based evaluation")
    logger.info(f"Task config: {cfg.task.task_config}")
    logger.info(f"Model: {cfg.model}")

    # Check if results already exist
    existing_results = list(output_path.rglob("results_*.json"))
    if existing_results:
        if cfg.override:
            logger.warning(
                f"Results already exist at {output_path}. Overriding due to --override flag."
            )
        else:
            logger.info(f"Results already exist at {output_path}")
            logger.info(f"Found {len(existing_results)} existing result files")
            logger.info("Skipping evaluation. Delete the directory to re-run.")
            return

    try:
        # Run evaluation

        evaluator = choose_task_evaluator(cfg.task.type)(
            model_wrapper=cfg.generation.model_wrapper,
            config_file_path=cfg.task.task_config,
            model_name=cfg.model.name,
            output_path=output_path,
            device=cfg.model.device,
            dtype=cfg.model.dtype,
            use_lmeval_format=True,
            n_iterations=cfg.task.nsolutions,
        )

        metrics = run_evaluation(
            evaluator,
            limit=cfg.limit,
            batch_size=cfg.generation.batch_size,
            seed=cfg.generation.seed,
            max_new_tokens=cfg.generation.max_gen_toks,
            use_chat_template=not cfg.template.disable_chat_template,
            enable_thinking=not cfg.template.disable_thinking,
        )

        logger.info("Evaluation completed successfully!")
        logger.info(f"Final metrics: {metrics}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
