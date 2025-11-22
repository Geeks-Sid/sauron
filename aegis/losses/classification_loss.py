import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    Args:
        alpha (float, list, torch.Tensor, optional): Weights for each class. 
            If float, it's treated as the weight for the rare class (binary).
            If list/Tensor, it should have length equal to number of classes.
        gamma (float): Focusing parameter. Higher values focus more on hard examples. Default: 2.0.
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'.
        label_smoothing (float): Label smoothing factor (0.0 to 1.0). Default: 0.0.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [Batch, Num_Classes]
            targets: Class indices [Batch]
        """
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
        
        # 1. Compute p_target for Focal Weight (based on true class)
        probs = F.softmax(inputs, dim=-1)
        p_target = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        focal_weight = (1 - p_target) ** self.gamma
        
        # 2. Compute Cross Entropy Loss (with optional label smoothing and class weights)
        # F.cross_entropy handles the class weights (alpha) and label smoothing internally
        loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha, label_smoothing=self.label_smoothing)
            
        # 3. Apply Focal Weight
        focal_loss = focal_weight * loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
