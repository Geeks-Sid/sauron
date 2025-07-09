import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from sauron.mil_models.ABMIL import ABMIL
from sauron.mil_models.TransMIL import TransMIL
from sauron.mil_models.MaxMIL import MaxMIL
from sauron.mil_models.MeanMIL import MeanMIL
from sauron.mil_models.MambaMIL import MambaMIL, MambaMIL_L, MambaMIL_XL
from sauron.mil_models.S4MIL import S4MIL, S4MIL_L, S4MIL_XL
from sauron.mil_models.WIKGMIL import WIKGMIL
from sauron.mil_models.DiffABMIL import DiffABMIL
from sauron.feature_extraction.models.patch_encoders.factory import encoder_factory


def mil_model_factory(args, in_dim=None):
    if in_dim is None:
        in_dim = encoder_factory(args.backbone).embedding_dim
    if args.mil_model == 'abmil':
        return ABMIL(in_dim=in_dim, n_classes=args.n_classes)
    elif args.mil_model == 'transmil':
        return TransMIL(in_dim=in_dim, n_classes=args.n_classes)
    elif args.mil_model == 'maxmil':
        return MaxMIL(in_dim=in_dim, n_classes=args.n_classes)
    elif args.mil_model == 'meanmil':
        return MeanMIL(in_dim=in_dim, n_classes=args.n_classes)
    elif args.mil_model == 'mamba':
        return MambaMIL(in_dim=in_dim, n_classes=args.n_classes)
    elif args.mil_model == 'mamba_l':
        return MambaMIL_L(in_dim=in_dim, n_classes=args.n_classes)
    elif args.mil_model == 'mamba_xl':
        return MambaMIL_XL(in_dim=in_dim, n_classes=args.n_classes)
    elif args.mil_model == 's4':
        return S4MIL(in_dim=in_dim, n_classes=args.n_classes)
    elif args.mil_model == 's4_l':
        return S4MIL_L(in_dim=in_dim, n_classes=args.n_classes)
    elif args.mil_model == 's4_xl':
        return S4MIL_XL(in_dim=in_dim, n_classes=args.n_classes)
    elif args.mil_model == 'wikgmil':
        return WIKGMIL(in_dim=in_dim, n_classes=args.n_classes)
    elif args.mil_model == 'diffabmil':
        return DiffABMIL(in_dim=in_dim, n_classes=args.n_classes)
    else:
        raise ValueError(f"Unknown MIL model: {args.mil_model}")