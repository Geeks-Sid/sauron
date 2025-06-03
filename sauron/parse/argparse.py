import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Configurations for Whole Slide Image (WSI) Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    argument_groups_config = [
        (
            "Data & I/O Configuration",
            [
                (
                    ("--data_root_dir",),
                    {
                        "type": str,
                        "default": None,
                        "help": "Specify the root directory where the dataset is located. This is essential for loading the data correctly.",
                    },
                ),
                (
                    ("--results_dir",),
                    {
                        "default": "./results",
                        "help": "Path to the directory where training results and model checkpoints will be saved. Default is './results'.",
                    },
                ),
                (
                    ("--split_dir",),
                    {
                        "type": str,
                        "default": None,
                        "help": "Path to the directory containing custom data splits. If not provided, splits will be generated based on the task and label fraction.",
                    },
                ),
                (
                    ("--patch_size",),
                    {
                        "type": str,
                        "default": "",
                        "help": "Define the size of image patches in the format [height]x[width]. This is important for processing images.",
                    },
                ),
                (
                    ("--resolution",),
                    {
                        "type": str,
                        "default": "20x",
                        "help": "Set the magnification level for processing images. Examples include '10x' or '10x_40x' for combined levels.",
                    },
                ),
                (
                    ("--early_fusion",),
                    {
                        "action": "store_true",
                        "default": False,
                        "help": "Enable or disable early fusion for models that utilize multiple magnification levels. This can enhance model performance.",
                    },
                ),
                (
                    ("--preloading",),
                    {
                        "choices": ["yes", "no"],
                        "default": "no",
                        "help": "Specify whether to preload data into memory for faster access during training. Options are 'yes' or 'no'.",
                    },
                ),
            ],
        ),
        (
            "Training Hyperparameters",
            [
                (
                    ("--max_epochs",),
                    {
                        "type": int,
                        "default": 200,
                        "help": "Set the maximum number of epochs for training the model. Default is 200.",
                    },
                ),
                (
                    ("--lr",),
                    {
                        "type": float,
                        "default": 1e-4,
                        "help": "Initial learning rate for the optimizer. Adjust this for better convergence.",
                    },
                ),
                (
                    ("--reg",),
                    {
                        "type": float,
                        "default": 1e-5,
                        "help": "Weight decay factor for L2 regularization. Helps prevent overfitting.",
                    },
                ),
                (
                    ("--opt",),
                    {
                        "choices": ["adam", "sgd", "adamw"],
                        "default": "adam",
                        "help": "Choose the optimizer to use for training. Options include 'adam', 'sgd', or 'adamw'.",
                    },
                ),
                (
                    ("--drop_out",),
                    {
                        "type": float,
                        "default": 0.25,
                        "help": "Set the dropout probability to prevent overfitting during training.",
                    },
                ),
                (
                    ("--early_stopping",),
                    {
                        "action": "store_true",
                        "help": "Enable early stopping to halt training when validation performance stops improving.",
                    },
                ),
                (
                    ("--weighted_sample",),
                    {
                        "action": "store_true",
                        "help": "Enable weighted sampling to address class imbalance in the training dataset.",
                    },
                ),
                (
                    ("--batch_size",),
                    {
                        "type": int,
                        "default": 1,
                        "help": "Set the batch size for training.",
                    },
                ),
            ],
        ),
        (
            "Model Configuration",
            [
                (
                    ("--model_type",),
                    {
                        "type": str,
                        "default": "att_mil",
                        "help": "Specify the type of model architecture to use for training. Default is 'att_mil'.",
                    },
                ),
                (
                    ("--backbone",),
                    {
                        "type": str,
                        "default": "resnet50",
                        "help": "Select the backbone network for feature extraction. Default is 'resnet50'.",
                    },
                ),
                (
                    ("--in_dim",),
                    {
                        "type": int,
                        "default": 1024,
                        "help": "Set the input dimension for the model. This should match the output of the backbone network.",
                    },
                ),
            ],
        ),
        (
            "MambaMIL Specific Configuration",  # Conditionally relevant if model_type is MambaMIL
            [
                (
                    ("--mambamil_rate",),
                    {
                        "type": int,
                        "default": 10,
                        "help": "Rate parameter for MambaMIL, influencing the model's behavior.",
                    },
                ),
                (
                    ("--mambamil_layer",),
                    {
                        "type": int,
                        "default": 2,
                        "help": "Number of layers in the MambaMIL architecture.",
                    },
                ),
                (
                    ("--mambamil_type",),
                    {
                        "choices": ["Mamba", "BiMamba", "SRMamba"],
                        "default": "SRMamba",
                        "help": "Select the type of Mamba architecture to use. Options include 'Mamba', 'BiMamba', or 'SRMamba'.",
                    },
                ),
            ],
        ),
        (
            "Experiment & Reproducibility",
            [
                (
                    ("--task",),
                    {
                        "type": str,
                        "required": True,
                        "help": "Specify the task name or identifier for the experiment.",
                    },
                ),
                (
                    ("--exp_code",),
                    {
                        "type": str,
                        "required": True,
                        "help": "Provide a unique experiment code for tracking purposes.",
                    },
                ),
                (
                    ("--seed",),
                    {
                        "type": int,
                        "default": 1,
                        "help": "Set the random seed for reproducibility of results. Default is 1.",
                    },
                ),
                (
                    ("--label_frac",),
                    {
                        "type": float,
                        "default": 1.0,
                        "help": "Specify the fraction of training labels to use. Default is 1.0 (use all labels).",
                    },
                ),
                (
                    ("--log_data",),
                    {
                        "action": "store_true",
                        "help": "Enable logging of training data using TensorBoard for visualization and analysis.",
                    },
                ),
                (
                    ("--testing",),
                    {
                        "action": "store_true",
                        "help": "Enable testing/debugging mode for the experiment.",
                    },
                ),
            ],
        ),
        (
            "Cross-Validation Configuration",
            [
                (
                    ("--k",),
                    {
                        "type": int,
                        "default": 10,
                        "help": "Specify the total number of folds for cross-validation. Default is 10.",
                    },
                ),
                (
                    ("--k_start",),
                    {
                        "type": int,
                        "default": -1,
                        "help": "Set the starting fold for cross-validation. Use -1 for the last fold.",
                    },
                ),
                (
                    ("--k_end",),
                    {
                        "type": int,
                        "default": -1,
                        "help": "Set the ending fold for cross-validation. Use -1 for the first fold.",
                    },
                ),
            ],
        ),
        (
            "Survival Configuration",
            [
                (
                    ("--bag_loss",),
                    {
                        "type": str,
                        "choices": ["svm", "ce", "ce_surv", "nll_surv", "cox_surv"],
                        "default": "nll_surv",
                        "help": "Slide-level classification loss function (default: nll_surv).",
                    },
                ),
                (
                    ("--alpha_surv",),
                    {
                        "type": float,
                        "default": 0.0,
                        "help": "How much to weigh uncensored patients.",
                    },
                ),
                (
                    ("--lambda_reg",),
                    {
                        "type": float,
                        "default": 1e-4,
                        "help": "L1-Regularization Strength (Default 1e-4).",
                    },
                ),
                (
                    ("--inst_loss",),
                    {
                        "type": str,
                        "choices": ["svm", "ce", None],
                        "default": None,
                        "help": "Instance-level clustering loss function (default: None).",
                    },
                ),
                (
                    ("--subtyping",),
                    {
                        "action": "store_true",
                        "default": False,
                        "help": "Enable subtyping problem.",
                    },
                ),
                (
                    ("--bag_weight",),
                    {
                        "type": float,
                        "default": 0.7,
                        "help": "Weight coefficient for bag-level loss (default: 0.7).",
                    },
                ),
                (
                    ("--B",),
                    {
                        "type": int,
                        "default": 8,
                        "help": "Number of positive/negative patches to sample for clam.",
                    },
                ),
                (
                    ("--gc",),
                    {
                        "type": int,
                        "default": 32,
                        "help": "Gradient Accumulation Step.",
                    },
                ),
            ],
        ),
    ]

    # Correcting the `early_fusion` argument based on best practices
    # Find and update early_fusion: argparse `type=bool` is problematic.
    # It's better to use `action='store_true'` or `action='store_false'`.
    # If `default=False`, use `action='store_true'`.
    for group_name, arg_list in argument_groups_config:
        for i, (name_or_flags, kwargs) in enumerate(arg_list):
            if name_or_flags == ("--early_fusion",):
                if "type" in kwargs and kwargs["type"] is bool:
                    del kwargs["type"]  # remove type=bool
                    if "default" in kwargs and kwargs["default"] is False:
                        kwargs["action"] = "store_true"
                    elif "default" in kwargs and kwargs["default"] is True:
                        kwargs["action"] = "store_false"
                    else:  # Default to store_true if not specified or ambiguous
                        kwargs["action"] = "store_true"
                    arg_list[i] = (name_or_flags, kwargs)  # Update the list
                break  # Found it

    for group_name, arg_definitions in argument_groups_config:
        group = parser.add_argument_group(group_name)
        for name_or_flags, kwargs in arg_definitions:
            group.add_argument(*name_or_flags, **kwargs)

    args = parser.parse_args()
    return args
