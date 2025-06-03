import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableAttentionMIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout=False, act="relu", num_heads=4):
        super(DifferentiableAttentionMIL, self).__init__()
        self.L = 512  # Embedding dimension
        self.D = 128  # Hidden dimension
        self.K = 1  # Number of attention heads (can be set to num_heads if needed)
        self.num_heads = num_heads
        self.head_dim = self.L // self.num_heads

        # Feature extraction layer
        layers = [nn.Linear(in_dim, self.L)]
        if act.lower() == "gelu":
            layers.append(nn.GELU())
        else:
            layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(0.25))
        self.feature = nn.Sequential(*layers)

        # Query, Key, and Value projections
        self.q_proj = nn.Linear(self.L, self.L)
        self.k_proj = nn.Linear(self.L, self.L)
        self.v_proj = nn.Linear(self.L, self.L)

        # Output projection
        self.out_proj = nn.Linear(self.L, self.L)

        # Classifier
        self.classifier = nn.Linear(self.L, n_classes)

    def forward(self, x):
        # x: (batch_size, num_instances, in_dim)
        batch_size, num_instances, _ = x.size()
        features = self.feature(x)  # (batch_size, num_instances, L)

        # Compute Q, K, V projections
        q = self.q_proj(features)  # (batch_size, num_instances, L)
        k = self.k_proj(features)  # (batch_size, num_instances, L)
        v = self.v_proj(features)  # (batch_size, num_instances, L)

        # Reshape for multi-head attention
        q = q.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Prepare for scaled_dot_product_attention
        q = q.reshape(batch_size * self.num_heads, num_instances, self.head_dim)
        k = k.reshape(batch_size * self.num_heads, num_instances, self.head_dim)
        v = v.reshape(batch_size * self.num_heads, num_instances, self.head_dim)

        # Compute attention using scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        # Reshape back to original dimensions
        attn_output = attn_output.view(
            batch_size, self.num_heads, num_instances, self.head_dim
        ).transpose(1, 2)
        attn_output = attn_output.contiguous().view(batch_size, num_instances, self.L)

        # Aggregate over instances (e.g., mean pooling)
        bag_repr = attn_output.mean(dim=1)  # (batch_size, L)

        # Classification
        logits = self.classifier(bag_repr)  # (batch_size, n_classes)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1)

        return logits, Y_prob, Y_hat, None, {}
