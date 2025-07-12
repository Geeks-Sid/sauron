# Import the refactored main functions from their new locations
from aegis.feature_extraction.cli_runner import run_feature_extraction_job

# IMPORTANT: Ensure aegis/parse/argparse.py is renamed to aegis/parse/cli_parsers.py
from aegis.parse.cli_parsers import (
    get_mil_args,
    parse_feature_extraction_arguments,
)
from aegis.training.cli_runner import run_mil_training_job


def feature_extract_main():
    """
    Entry point for the 'aegis-extract' command.
    Parses arguments and runs the feature extraction pipeline.
    """
    args = parse_feature_extraction_arguments()
    print(f"Launching aegis Feature Extraction with arguments: {args}")
    run_feature_extraction_job(args)
    print("aegis Feature Extraction job completed.")


def train_mil_main():
    """
    Entry point for the 'aegis-train' command.
    Parses arguments and runs the MIL training pipeline.
    """
    args = get_mil_args()
    # The `train_mil.py` script included some argument compatibility logic,
    # replicate it here if `get_mil_args` doesn't fully handle it.
    if not hasattr(args, "task_name"):
        args.task_name = args.task
    if not hasattr(args, "k_fold"):
        args.k_fold = args.k

    print(f"Launching aegis MIL Training with arguments: {args}")
    run_mil_training_job(args)
    print("aegis MIL Training job completed.")


if __name__ == "__main__":
    # This block allows for direct testing of `aegis/cli.py` during development.
    # In a real installation, setuptools handles calling `feature_extract_main` or `train_mil_main`.

    # Example for direct testing:
    # Set sys.argv to simulate command-line arguments
    # sys.argv = ["cli.py", "feature_extract", "--job_dir", "./test_job", "--wsi_dir", "./test_wsi", "--task", "cache", "--gpu", "0"]
    # feature_extract_main()

    # sys.argv = ["cli.py", "train_mil", "--task", "my_cls_task", "--task_type", "classification", "--exp_code", "test_exp", "--data_root_dir", "./data", "--dataset_csv", "./data/dataset.csv"]
    # train_mil_main()

    print("To run, install the package and use 'aegis-extract' or 'aegis-train'.")
    print("For help, run 'aegis-extract --help' or 'aegis-train --help'.")
