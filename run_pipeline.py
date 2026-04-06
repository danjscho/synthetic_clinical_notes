"""
run_pipeline.py — production entrypoint for the synthetic clinical notes pipeline.

Usage:
    python run_pipeline.py [options]

Options:
    --generations N      Number of patient journeys to generate (overrides params.py)
    --model NAME              LLM model name served by the local endpoint (overrides params.py)
    --validation-model NAME   LLM model for validation steps (overrides params.py, defaults to --model)
    --data-dir PATH      Directory to read input CSV files from (overrides config.py)
    --output-dir PATH    Directory to write output CSV files to (overrides config.py)
    --test-mode          Generate one clinical note per patient (overrides params.py)
    --run-name NAME      Label for this run, included in output metadata
"""

import argparse
import asyncio
import logging
import pathlib
import sys
from datetime import datetime

import urllib.request
import urllib.error


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic clinical notes pipeline")
    parser.add_argument("--generations", type=int, help="Number of patient journeys to generate")
    parser.add_argument("--model", type=str, help="LLM model name (must match what the server is serving)")
    parser.add_argument("--validation-model", type=str, help="LLM model for validation steps (defaults to --model if not set)")
    parser.add_argument("--data-dir", type=str, help="Directory to read input CSV files from")
    parser.add_argument("--output-dir", type=str, help="Directory to write output CSV files to")
    parser.add_argument("--test-mode", action="store_true", help="Generate one clinical note per patient")
    parser.add_argument("--run-name", type=str, default="", help="Label for this run")
    parser.add_argument("--concurrency", type=int, help="Max concurrent LLM calls (default: from params.py, recommended 2-4 for local models)")
    parser.add_argument("--resume", action="store_true", help="Resume a previously interrupted run: skip completed patients and append to checkpoint files")
    parser.add_argument("--evaluate", action="store_true", help="Run quality evaluation (fluency, groundedness, relevance, readability) after the pipeline and write evaluation_results.csv")
    return parser.parse_args()


def configure_logging(log_dir: pathlib.Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger(__name__), log_file


def apply_config_overrides(args):
    """Apply CLI args to config modules before any pipeline imports read them."""
    import config.config as cfg
    from config.params import PARAMS

    if args.data_dir:
        cfg.DATA_DIR = args.data_dir
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.model:
        PARAMS["pipeline_config"]["model"] = args.model
    if args.validation_model:
        PARAMS["pipeline_config"]["validation_model"] = args.validation_model
    if args.generations:
        PARAMS["pipeline_config"]["number_of_generations"] = args.generations
    if args.test_mode:
        PARAMS["pipeline_config"]["TEST_MODE"] = True
    if args.concurrency:
        PARAMS["pipeline_config"]["llm_concurrency"] = args.concurrency
    if args.resume:
        PARAMS["pipeline_config"]["resume"] = True
    if args.evaluate:
        PARAMS["pipeline_config"]["evaluate"] = True


def preflight_check(logger):
    """Validate environment before any LLM calls. Exits with code 1 on failure."""
    import config.config as cfg
    from config.params import PARAMS

    ok = True

    # 1. LLM endpoint reachable
    health_url = cfg.LLM_BASE_URL.rstrip("/").removesuffix("/v1") + "/"
    try:
        urllib.request.urlopen(health_url, timeout=5)
        logger.info(f"LLM endpoint reachable: {health_url}")
    except urllib.error.URLError as e:
        logger.error(f"LLM endpoint not reachable at {health_url}: {e}")
        ok = False

    # 2. Required input CSVs exist
    data_dir = pathlib.Path(cfg.DATA_DIR)
    required_csvs = [
        PARAMS["pipeline_config"]["patients_input_dataset"],
        PARAMS["pipeline_config"]["emergency_admissions_dataset"],
        PARAMS["pipeline_config"]["elective_admissions_dataset"],
    ]
    for name in required_csvs:
        path = data_dir / f"{name}.csv"
        if path.exists():
            logger.info(f"Input CSV found: {path}")
        else:
            logger.error(f"Input CSV missing: {path}")
            ok = False

    # 3. Output dir writable
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        logger.info(f"Output dir writable: {output_dir}")
    except OSError as e:
        logger.error(f"Output dir not writable at {output_dir}: {e}")
        ok = False

    if not ok:
        logger.error("Preflight check failed — fix the above errors before running the pipeline.")
        sys.exit(1)

    logger.info("Preflight check passed.")


async def main(args, logger):
    # Pipeline stage imports are deferred so config overrides (applied before main()
    # is called) are visible when data_generator reads PARAMS at module level.
    from src.data_generator import (
        generate_patients,
        generate_admissions,
        generate_journeys,
        generate_clinical_notes,
        add_augmentations,
        save_final_outputs,
    )
    from src.dataset_utils import write_dataset
    from src.evaluation_utils import run_evaluation
    from config.params import PARAMS

    current_time = datetime.now()
    run_name = args.run_name
    version_tag = current_time.strftime("%Y-%m-%d") + (f"/{run_name}" if run_name else "")

    logger.info(f"Run name: {run_name!r}  |  Version tag: {version_tag}")

    # --- Stage 1: Patients ---
    logger.info("=== Stage: Patients ===")
    try:
        patient_generator = generate_patients()
        await patient_generator.run(return_output=True)
        patient_generator.write_patients_to_dataset()
    except Exception as e:
        logger.error(f"Stage 'Patients' failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Stage 2: Admissions ---
    logger.info("=== Stage: Admissions ===")
    try:
        admission_generator = generate_admissions()
        await admission_generator.run(return_output=True)
        admission_generator.write_admissions_to_dataset()
    except Exception as e:
        logger.error(f"Stage 'Admissions' failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Stage 3: Journeys ---
    logger.info("=== Stage: Journeys ===")
    try:
        journey_generator = generate_journeys()
        await journey_generator.run(return_outputs=True)
        journey_generator.write_journeys_to_dataset()
    except Exception as e:
        logger.error(f"Stage 'Journeys' failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Stage 4: Clinical Notes ---
    logger.info("=== Stage: Clinical Notes ===")
    try:
        clinical_note_generator = generate_clinical_notes()
        await clinical_note_generator.run(return_output=True)
        clinical_note_generator.write_patient_documents_to_dataset()
    except Exception as e:
        logger.error(f"Stage 'Clinical Notes' failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Stage 5: Augmentations ---
    logger.info("=== Stage: Augmentations ===")
    try:
        augmentator = add_augmentations()
        await augmentator.run(True)
        augmentator.write_final_documents_to_dataset()
    except Exception as e:
        logger.error(f"Stage 'Augmentations' failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Stage 6: Final Output ---
    logger.info("=== Stage: Final Output ===")
    try:
        output_saver = save_final_outputs()
        output_saver.run(run_name, current_time, version_tag)
    except Exception as e:
        logger.error(f"Stage 'Final Output' failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Optional: Evaluation ---
    if PARAMS["pipeline_config"].get("evaluate", False):
        logger.info("=== Stage: Evaluation ===")
        try:
            from src.processing import _current_stage, log_stage_summary
            _current_stage.set("evaluation")
            evaluation_df = await run_evaluation(output_saver.evaluation_output_data)
            write_dataset(evaluation_df, "evaluation_results")
            log_stage_summary("evaluation")
            logger.info(f"Evaluation complete: {len(evaluation_df)} notes scored.")
        except Exception as e:
            logger.error(f"Stage 'Evaluation' failed: {e}", exc_info=True)
            sys.exit(1)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    args = parse_args()

    logger, log_file = configure_logging(pathlib.Path("logs"))
    logger.info(f"Logging to {log_file}")

    apply_config_overrides(args)

    preflight_check(logger)

    asyncio.run(main(args, logger))
