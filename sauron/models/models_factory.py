from typing import Any

import torch.nn as nn


# --- Model Initialization ---
def initialize_mil_model(args: Any) -> nn.Module:
    """Initialize the MIL model based on args namespace."""
    from sauron.mil_models.ABMIL import DAttention
    from sauron.mil_models.MambaMIL import MambaMIL
    from sauron.mil_models.MaxMIL import MaxMIL
    from sauron.mil_models.MeanMIL import MeanMIL
    from sauron.mil_models.S4MIL import S4Model
    from sauron.mil_models.TransMIL import TransMIL

    in_dim = args.in_dim
    n_classes = args.n_classes
    dropout = args.drop_out
    act = getattr(
        args, "act_fn", "gelu"
    )  # Allow overriding activation, default to gelu

    # task_type should be definitively set in args by the parser
    is_survival = args.task_type == "survival"

    model_type = args.model_type.lower()
    print(
        f"Initializing model: {model_type} (Activation: {act}, Survival: {is_survival})"
    )

    model_params = {
        "in_dim": in_dim,
        "n_classes": n_classes,
        "dropout": dropout,
        "act": act,
        "survival": is_survival,
    }

    if model_type == "mean_mil":
        model = MeanMIL(**model_params)
    elif model_type == "max_mil":
        model = MaxMIL(**model_params)
    elif model_type == "att_mil":
        model = DAttention(**model_params)
    elif model_type == "trans_mil":
        model = TransMIL(**model_params)
    elif model_type in ["s4_mil", "s4model"]:
        model = S4Model(**model_params)
    elif model_type == "mamba_mil":
        model = MambaMIL(
            **model_params,
            layer=args.mambamil_layer,
            rate=args.mambamil_rate,
            type=args.mambamil_type,
        )
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented.")

    # if hasattr(args, 'init_weights') and args.init_weights:
    #     model.apply(initialize_weights) # Define initialize_weights if used
    return model
