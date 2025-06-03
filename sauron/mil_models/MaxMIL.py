import torch
import torch.nn as nn
import torch.nn.functional as F

from sauron.utils.generic_utils import initialize_weights


class MaxMIL(nn.Module):
    def __init__(
        self, in_dim=1024, n_classes=1, dropout=True, act="relu", survival=False
    ):
        super(MaxMIL, self).__init__()

        head = [nn.Linear(in_dim, 512)]

        if act.lower() == "relu":
            head += [nn.ReLU()]
        elif act.lower() == "gelu":
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
        head += [nn.Linear(512, n_classes)]
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

        self.survival = survival

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[0] > 1:
            raise RuntimeError(
                "Batch size must be 1, current batch size is:{}".format(x.shape[0])
            )
        if len(x.shape) == 3 and x.shape[0] == 1:
            x = x[0]

        logits = self.head(x)
        logits, _ = torch.max(logits, dim=0, keepdim=True)

        """
        Survival Layer
        """
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None
        return logits, Y_prob, Y_hat, A_raw, results_dict

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = self.head.to(device)
