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

class Poly1Loss(nn.Module):
    """
    Poly1Loss: A alternative to Focal Loss that adds a polynomial term to Cross Entropy.
    Often provides better stability and accuracy than Focal Loss by preventing 
    over-suppression of the gradient for easy samples.
    
    Formula: L_Poly1 = L_CE + epsilon * (1 - Pt)
    
    Args:
        num_classes (int): Number of classes.
        epsilon (float): Weight for the polynomial term. Default: 1.0.
        reduction (str): 'mean', 'sum', or 'none'.
        weight (torch.Tensor, optional): Class weights.
    """
    def __init__(self, num_classes, epsilon=1.0, reduction='mean', weight=None):
        super(Poly1Loss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        if self.weight is not None and self.weight.device != inputs.device:
            self.weight = self.weight.to(inputs.device)

        # 1. Standard Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        
        # 2. Polynomial term (1 - Pt)
        pt = F.softmax(inputs, dim=-1)
        p_target = pt.gather(1, targets.view(-1, 1)).squeeze(1)
        poly1 = self.epsilon * (1 - p_target)
        
        # 3. Combine
        loss = ce_loss + poly1
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
