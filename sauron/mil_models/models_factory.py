from typing import Any, Dict, Type, Union

import torch.nn as nn

from .ABMIL import DAttention as AttMIL  # Using DAttention as att_mil
from .ABMIL import GatedAttention
from .DiffABMIL import DifferentiableAttentionMIL
from .MambaMIL import MambaMIL
from .MaxMIL import MaxMIL
from .MeanMIL import MeanMIL
from .S4MIL import S4Model
from .TransMIL import TransMIL
from .WIKGMIL import WiKG

# Mapping of model types to their corresponding classes
MODEL_MAP: Dict[str, Type[nn.Module]] = {
    "mean_mil": MeanMIL,
    "max_mil": MaxMIL,
    "att_mil": AttMIL,  # Standard Attention MIL (ABMIL's DAttention)
    "gated_att_mil": GatedAttention,  # Gated Attention MIL
    "diff_abmil": DifferentiableAttentionMIL,  # Differentiable Attention (Multi-Head)
    "trans_mil": TransMIL,
    "s4_mil": S4Model,  # Alias "s4model" can be handled by .lower()
    "mamba_mil": MambaMIL,
    "wikg_mil": WiKG,
}


def initialize_mil_model(args: Any) -> nn.Module:
    """Initialize the MIL model based on args namespace."""

    model_type_key = args.model_type.lower()

    # Common parameters for most models
    common_params = {
        "in_dim": args.in_dim,
        "n_classes": args.n_classes,
        "dropout_rate": getattr(
            args, "dropout_rate", getattr(args, "drop_out", 0.25)
        ),  # Standardize dropout arg name
        "activation": getattr(
            args, "activation_fn", "relu"
        ),  # Standardize activation arg name
        "is_survival": args.task_type == "survival",
    }

    print(
        f"Initializing model: {model_type_key} (Activation: {common_params['activation']}, "
        f"Survival: {common_params['is_survival']}, Dropout: {common_params['dropout_rate']})"
    )

    if model_type_key not in MODEL_MAP:
        raise NotImplementedError(
            f"Model type '{model_type_key}' not implemented or not in MODEL_MAP."
        )

    model_class = MODEL_MAP[model_type_key]

    # Model-specific parameters
    specific_params = {}
    if model_type_key == "mamba_mil":
        specific_params.update(
            {
                "embed_dim": getattr(args, "mambamil_embed_dim", 512),
                "num_mamba_layers": getattr(
                    args, "mambamil_layers", 2
                ),  # Renamed from mambamil_layer
                "mamba_type": getattr(args, "mambamil_type", "SRMamba"),
                "srmamba_rate": getattr(args, "mambamil_rate", 10),
            }
        )
    elif model_type_key == "s4_mil":
        specific_params.update(
            {
                "embed_dim": getattr(args, "s4mil_embed_dim", 512),
                "s4_d_state": getattr(args, "s4mil_d_state", 64),
            }
        )
    elif model_type_key == "trans_mil":
        specific_params.update(
            {
                "embed_dim": getattr(args, "transmil_embed_dim", 512),
                "num_transformer_layers": getattr(args, "transmil_layers", 2),
                "num_attn_heads": getattr(args, "transmil_heads", 8),
            }
        )
    elif model_type_key == "att_mil" or model_type_key == "gated_att_mil":
        # ABMIL variants might have specific embed_dim, attention_hidden_dim if configurable
        pass  # Uses default internal dimensions or can be made configurable via args
    elif model_type_key == "diff_abmil":
        specific_params.update(
            {
                "embed_dim": getattr(args, "diffabmil_embed_dim", 512),
                "num_heads": getattr(args, "diffabmil_num_heads", 8),
            }
        )
    elif model_type_key == "wikg_mil":
        specific_params.update(
            {
                "hidden_dim": getattr(args, "wikg_hidden_dim", 512),
                "top_k_neighbors": getattr(args, "wikg_topk", 6),
                "agg_type": getattr(args, "wikg_agg_type", "bi-interaction"),
                "pool_type": getattr(args, "wikg_pool_type", "attn"),
                # activation for WiKG is handled in common_params, but WiKG often uses LeakyReLU
                "activation": getattr(
                    args, "activation_fn", "leaky_relu"
                ),  # Default leaky for WiKG
            }
        )
    elif model_type_key in ["max_mil", "mean_mil"]:
        specific_params.update(
            {
                "hidden_dim": getattr(
                    args, f"{model_type_key.split('_')[0]}_mil_hidden_dim", 512
                ),
            }
        )

    # Combine common and specific parameters
    all_params = {**common_params, **specific_params}

    # Filter params for the specific model constructor
    # This requires inspecting the model's __init__ signature, or assuming all models handle extra **kwargs
    # For robustness, it's better to only pass expected args.
    # However, many __init__ are now standardized.
    # A simpler approach for now is to pass all_params and let models pick what they need
    # if their __init__ are designed for that or we ensure only relevant ones are passed.

    # For now, let's assume model constructors can handle the combined dict,
    # or we'll rely on the specific_params logic to be exhaustive for non-common ones.
    # A more robust way: inspect model_class.__init__ and filter.
    # For this refactor, I'm standardizing __init__ so common_params + specific_params should largely work.

    return model_class(**all_params)
