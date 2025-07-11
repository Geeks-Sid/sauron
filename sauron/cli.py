# Import the refactored main functions from their new locations
from sauron.feature_extraction.cli_runner import run_feature_extraction_job

# IMPORTANT: Ensure sauron/parse/argparse.py is renamed to sauron/parse/cli_parsers.py
from sauron.parse.cli_parsers import (
    get_mil_args,
    parse_feature_extraction_arguments,
)
from sauron.training.cli_runner import run_mil_training_job


def feature_extract_main():
    """
    Entry point for the 'sauron-extract' command.
    Parses arguments and runs the feature extraction pipeline.
    """
    args = parse_feature_extraction_arguments()
    print(f"Launching Sauron Feature Extraction with arguments: {args}")
    run_feature_extraction_job(args)
    print("Sauron Feature Extraction job completed.")


def train_mil_main():
    """
    Entry point for the 'sauron-train' command.
    Parses arguments and runs the MIL training pipeline.
    """
    args = get_mil_args()
    # The `train_mil.py` script included some argument compatibility logic,
    # replicate it here if `get_mil_args` doesn't fully handle it.
    if not hasattr(args, "task_name"):
        args.task_name = args.task
    if not hasattr(args, "k_fold"):
        args.k_fold = args.k

    print(f"Launching Sauron MIL Training with arguments: {args}")
    run_mil_training_job(args)
    print("Sauron MIL Training job completed.")


if __name__ == "__main__":
    # This block allows for direct testing of `sauron/cli.py` during development.
    # In a real installation, setuptools handles calling `feature_extract_main` or `train_mil_main`.

    # Example for direct testing:
    # Set sys.argv to simulate command-line arguments
    # sys.argv = ["cli.py", "feature_extract", "--job_dir", "./test_job", "--wsi_dir", "./test_wsi", "--task", "cache", "--gpu", "0"]
    # feature_extract_main()

    # sys.argv = ["cli.py", "train_mil", "--task", "my_cls_task", "--task_type", "classification", "--exp_code", "test_exp", "--data_root_dir", "./data", "--dataset_csv", "./data/dataset.csv"]
    # train_mil_main()

    print("To run, install the package and use 'sauron-extract' or 'sauron-train'.")
    print("For help, run 'sauron-extract --help' or 'sauron-train --help'.")
