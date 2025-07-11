import numpy as np
import torch


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(
            1 - hazards, dim=1
        )  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat(
        [torch.ones_like(c), S], 1
    )  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(
            1 - hazards, dim=1
        )  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y) + eps)
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    ce_l = -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(
        1 - torch.gather(S, 1, Y).clamp(min=eps)
    )
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


class CoxSurvLoss(object):
    def __call__(self, hazards, S, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data

        # Ensure hazards, S (event times), and c (censoring status) are 1D tensors
        # hazards: Predicted log-risk scores for each patient. Shape: [batch_size]
        # S: Observed event/censoring times for each patient. Shape: [batch_size]
        # c: Censoring status (0 for event, 1 for censored) for each patient. Shape: [batch_size]

        # Squeeze inputs to ensure they are 1D, as Cox loss operates on scalar times/risks per patient.
        hazards = hazards.squeeze()
        S = S.squeeze()
        c = c.squeeze()

        # Basic validation for input dimensions after squeezing
        if hazards.dim() != 1 or S.dim() != 1 or c.dim() != 1:
            raise ValueError(
                "Input tensors hazards, S, and c must be 1-dimensional after squeezing."
                f" Got hazards.shape={hazards.shape}, S.shape={S.shape}, c.shape={c.shape}"
            )

        current_batch_len = len(S)

        # R_mat[i, j] = 1 if patient j is in the risk set of patient i (i.e., S[j] >= S[i]), else 0.
        # This can be computed efficiently using broadcasting, removing the slow numpy loop.
        # S.unsqueeze(0) makes S a row vector (1, batch_size)
        # S.unsqueeze(1) makes S a column vector (batch_size, 1)
        # The comparison (S.unsqueeze(0) >= S.unsqueeze(1)) then broadcasts to (batch_size, batch_size)
        # where result[row_i, col_j] = (S[col_j] >= S[row_i]).
        R_mat = (S.unsqueeze(0) >= S.unsqueeze(1)).float()

        # Ensure R_mat is on the same device as the other tensors (hazards, S, c)
        device = hazards.device
        R_mat = R_mat.to(device)

        theta = (
            hazards  # hazards should already be the log-risk scores, now confirmed 1D
        )
        exp_theta = torch.exp(theta)

        # Calculate the log sum over the risk set for each patient i: log(sum_{j in R_i} exp(theta_j))
        # torch.sum(exp_theta * R_mat, dim=1) sums exp_theta_j for all j where R_mat[i,j] is 1 (i.e., j is in risk set of i)
        log_risk_sum = torch.log(torch.sum(exp_theta * R_mat, dim=1))

        # The Cox proportional hazards partial log-likelihood:
        # L = - sum_{i: uncensored} (theta_i - log(sum_{j in R_i} exp(theta_j)))
        # (1 - c) acts as a mask, making the term zero for censored observations.
        loss_cox = -torch.mean((theta - log_risk_sum) * (1 - c))
        return loss_cox
