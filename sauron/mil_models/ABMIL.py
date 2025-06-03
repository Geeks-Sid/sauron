import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sauron.utils.generic_utils import initialize_weights


class DAttention(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival=False):
        super(DAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(in_dim, 512)]
        self.survival = survival

        if act.lower() == "gelu":
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )

        # self.apply(initialize_weights)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.squeeze()
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A_raw = A
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL

        logits = self.classifier(M)

        """
        Survival layer
        """
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        # keep the same API with the clam
        return logits, Y_prob, Y_hat, A_raw, {}

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature = self.feature.to(device)
        self.attention = self.attention.to(device)
        self.classifier = self.classifier.to(device)


class GatedAttention(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival=False):
        super(GatedAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(in_dim, 512)]
        self.survival = survival
        if act.lower() == "gelu":
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.squeeze()

        A_V = self.attention_V(feature)  # NxD
        A_U = self.attention_U(feature)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, feature)  # KxL

        logits = self.classifier(M)

        """
        Survival layer
        """
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        # keep the same API with the clam
        return logits, Y_prob, Y_hat, None, {}
