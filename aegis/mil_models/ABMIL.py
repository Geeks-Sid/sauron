import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis.utils.generic_utils import initialize_weights  # Assuming this exists

from .activations import get_activation_fn


class _BaseAttentionMIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        attention_hidden_dim: int,
        num_attention_outputs: int,  # Typically K=1 for MIL aggregation
        n_classes: int,
        dropout_rate: float = 0.25,
        activation: str = "relu",
        is_survival: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.num_attention_outputs = num_attention_outputs  # K
        self.is_survival = is_survival
        self.n_classes = n_classes

        feature_extractor_layers = [nn.Linear(in_dim, self.embed_dim)]
        feature_extractor_layers.append(get_activation_fn(activation))
        if dropout_rate > 0:
            feature_extractor_layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        self.classifier_layer = nn.Linear(
            self.embed_dim * self.num_attention_outputs, n_classes
        )
        # self.apply(initialize_weights) # Apply if this is a common practice for all sub-modules

    def _get_outputs(self, logits, attention_scores=None):
        # Predictions (highest logit index)
        predictions = torch.topk(logits, 1, dim=1)[1]

        if self.is_survival:
            hazards = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return hazards, survival_curves, predictions, attention_scores, {}
        else:
            probabilities = F.softmax(logits, dim=1)
            return logits, probabilities, predictions, attention_scores, {}


class DAttention(
    nn.Module
):  # Original DAttention renamed for clarity if _BaseAttentionMIL is used
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        dropout_rate: float = 0.25,
        activation: str = "relu",
        is_survival: bool = False,
    ):
        super().__init__()
        self.embed_dim = 512  # L
        self.attention_hidden_dim = 128  # D
        self.num_attention_outputs = 1  # K

        self.is_survival = is_survival
        self.n_classes = n_classes

        feature_layers = [nn.Linear(in_dim, self.embed_dim)]
        feature_layers.append(get_activation_fn(activation))
        if dropout_rate > 0:
            feature_layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor = nn.Sequential(*feature_layers)

        self.attention_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.attention_hidden_dim, self.num_attention_outputs),
        )
        self.classifier = nn.Linear(
            self.embed_dim * self.num_attention_outputs, n_classes
        )
        self.apply(initialize_weights)

    def forward(self, input_tensor: torch.Tensor):
        # input_tensor: (batch_size, num_instances, in_dim)
        batch_size = input_tensor.shape[0]

        instance_features = self.feature_extractor(
            input_tensor
        )  # (batch_size, num_instances, embed_dim)

        attention_logits = self.attention_net(
            instance_features
        )  # (batch_size, num_instances, K)
        attention_logits = torch.transpose(
            attention_logits, 2, 1
        )  # (batch_size, K, num_instances)

        attention_scores = F.softmax(attention_logits, dim=2)  # Softmax over instances

        # M = KxL equivalent for batch: (batch_size, K, embed_dim)
        aggregated_features = torch.bmm(attention_scores, instance_features)

        # If K=1, aggregated_features is (batch_size, 1, embed_dim)
        # Flatten for classifier: (batch_size, K * embed_dim)
        aggregated_features_flat = aggregated_features.view(batch_size, -1)

        logits = self.classifier(aggregated_features_flat)  # (batch_size, n_classes)

        # Predictions (highest logit index)
        predictions = torch.topk(logits, 1, dim=1)[1]

        if self.is_survival:
            hazards = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return (
                hazards,
                survival_curves,
                predictions,
                attention_logits.transpose(2, 1),
                {},
            )  # Return raw attention before softmax
        else:
            probabilities = F.softmax(logits, dim=1)
            return (
                logits,
                probabilities,
                predictions,
                attention_logits.transpose(2, 1),
                {},
            )


class GatedAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        dropout_rate: float = 0.25,
        activation: str = "relu",
        is_survival: bool = False,
    ):
        super().__init__()
        self.embed_dim = 512  # L
        self.attention_hidden_dim = 128  # D
        self.num_attention_outputs = 1  # K

        self.is_survival = is_survival
        self.n_classes = n_classes

        feature_layers = [nn.Linear(in_dim, self.embed_dim)]
        feature_layers.append(get_activation_fn(activation))
        if dropout_rate > 0:
            feature_layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor = nn.Sequential(*feature_layers)

        self.attention_V = nn.Sequential(
            nn.Linear(self.embed_dim, self.attention_hidden_dim), nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.embed_dim, self.attention_hidden_dim), nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(
            self.attention_hidden_dim, self.num_attention_outputs
        )

        self.classifier = nn.Linear(
            self.embed_dim * self.num_attention_outputs, n_classes
        )
        self.apply(initialize_weights)

    def forward(self, input_tensor: torch.Tensor):
        # input_tensor: (batch_size, num_instances, in_dim)
        batch_size = input_tensor.shape[0]

        instance_features = self.feature_extractor(
            input_tensor
        )  # (batch_size, num_instances, embed_dim)

        attention_values = self.attention_V(
            instance_features
        )  # (batch_size, num_instances, D)
        attention_units = self.attention_U(
            instance_features
        )  # (batch_size, num_instances, D)

        # Element-wise multiplication, then pass through weights layer
        unnormalized_attention_scores = self.attention_weights(
            attention_values * attention_units
        )  # (batch_size, num_instances, K)
        unnormalized_attention_scores = torch.transpose(
            unnormalized_attention_scores, 2, 1
        )  # (batch_size, K, num_instances)

        normalized_attention_scores = F.softmax(
            unnormalized_attention_scores, dim=2
        )  # Softmax over instances

        aggregated_features = torch.bmm(
            normalized_attention_scores, instance_features
        )  # (batch_size, K, embed_dim)
        flattened_aggregated_features = aggregated_features.view(
            batch_size, -1
        )  # (batch_size, K * embed_dim)

        logits = self.classifier(
            flattened_aggregated_features
        )  # (batch_size, n_classes)

        # Predictions (highest logit index)
        predictions = torch.topk(logits, 1, dim=1)[1]

        if self.is_survival:
            hazard_rates = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazard_rates, dim=1)
            # A_raw for GatedAttention is less direct, similar to DAttention, returning unnormalized_attention_scores before softmax
            return (
                hazard_rates,
                survival_curves,
                predictions,
                unnormalized_attention_scores.transpose(2, 1),
                {},
            )
        else:
            probabilities = F.softmax(logits, dim=1)
            return (
                logits,
                probabilities,
                predictions,
                unnormalized_attention_scores.transpose(2, 1),
                {},
            )
