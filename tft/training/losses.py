"""
Loss functions for Temporal Fusion Transformer.

This module implements the quantile loss (pinball loss) used for
probabilistic forecasting in TFT.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) for probabilistic forecasting.

    The quantile loss is asymmetric and penalizes under-prediction
    and over-prediction differently based on the quantile value.

    For a quantile q:
        L_q(y, ŷ) = (y - ŷ) * q      if y >= ŷ  (under-prediction)
                    (ŷ - y) * (1-q)   if y < ŷ   (over-prediction)

    Args:
        quantiles: List of quantiles to compute loss for (e.g., [0.1, 0.5, 0.9])
        reduction: Reduction method ('mean', 'sum', or 'none')
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        reduction: str = 'mean',
    ):
        super().__init__()

        self.quantiles = quantiles
        self.reduction = reduction

        # Validate quantiles
        for q in quantiles:
            assert 0 < q < 1, f"Quantile {q} must be in (0, 1)"

        # Register quantiles as buffer (not a parameter, but part of state_dict)
        self.register_buffer(
            'quantile_values',
            torch.tensor(quantiles, dtype=torch.float32),
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute quantile loss.

        Args:
            predictions: Predicted quantiles of shape (batch_size, seq_len, num_quantiles)
            targets: Ground truth values of shape (batch_size, seq_len)

        Returns:
            Quantile loss (scalar if reduction != 'none')
        """
        # Expand targets to match predictions
        # (batch_size, seq_len) -> (batch_size, seq_len, num_quantiles)
        targets = targets.unsqueeze(-1).expand_as(predictions)

        # Compute errors
        errors = targets - predictions

        # Expand quantile values for broadcasting
        # (num_quantiles,) -> (1, 1, num_quantiles)
        quantiles = self.quantile_values.view(1, 1, -1)

        # Compute asymmetric loss
        # When error > 0 (under-prediction): loss = error * quantile
        # When error < 0 (over-prediction): loss = -error * (1 - quantile)
        loss = torch.max(
            (quantiles - 1) * errors,
            quantiles * errors,
        )

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class NormalizedQuantileLoss(nn.Module):
    """
    Normalized Quantile Loss.

    Divides the quantile loss by the scale of the target variable
    to make it scale-independent.

    Args:
        quantiles: List of quantiles
        reduction: Reduction method
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        reduction: str = 'mean',
    ):
        super().__init__()

        self.quantile_loss = QuantileLoss(quantiles=quantiles, reduction='none')
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute normalized quantile loss.

        Args:
            predictions: Predicted quantiles
            targets: Ground truth values
            scale: Optional scaling factor (if None, uses target mean)

        Returns:
            Normalized quantile loss
        """
        # Compute quantile loss
        loss = self.quantile_loss(predictions, targets)

        # Compute scale if not provided
        if scale is None:
            scale = torch.abs(targets).mean() + 1e-8  # Add epsilon to avoid division by zero

        # Normalize by scale
        normalized_loss = loss / scale

        # Apply reduction
        if self.reduction == 'mean':
            return normalized_loss.mean()
        elif self.reduction == 'sum':
            return normalized_loss.sum()
        else:  # 'none'
            return normalized_loss


class WeightedQuantileLoss(nn.Module):
    """
    Weighted Quantile Loss.

    Allows different weights for different quantiles.
    Useful for emphasizing certain quantiles (e.g., median).

    Args:
        quantiles: List of quantiles
        weights: List of weights for each quantile (if None, all equal)
        reduction: Reduction method
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        weights: Optional[List[float]] = None,
        reduction: str = 'mean',
    ):
        super().__init__()

        self.quantile_loss = QuantileLoss(quantiles=quantiles, reduction='none')
        self.reduction = reduction

        if weights is None:
            weights = [1.0] * len(quantiles)

        assert len(weights) == len(quantiles), \
            "Number of weights must match number of quantiles"

        self.register_buffer(
            'weights',
            torch.tensor(weights, dtype=torch.float32),
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted quantile loss.

        Args:
            predictions: Predicted quantiles
            targets: Ground truth values

        Returns:
            Weighted quantile loss
        """
        # Compute quantile loss
        # (batch_size, seq_len, num_quantiles)
        loss = self.quantile_loss(predictions, targets)

        # Apply weights
        # (1, 1, num_quantiles)
        weights = self.weights.view(1, 1, -1)
        weighted_loss = loss * weights

        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:  # 'none'
            return weighted_loss


def quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """
    Functional form of quantile loss for a single quantile.

    Args:
        predictions: Predictions
        targets: Ground truth
        quantile: Quantile value (0 < q < 1)

    Returns:
        Quantile loss
    """
    errors = targets - predictions
    loss = torch.max(
        (quantile - 1) * errors,
        quantile * errors,
    )
    return loss.mean()


def compute_quantile_coverage(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile: float,
) -> float:
    """
    Compute empirical coverage of a quantile.

    For a well-calibrated model, the coverage should be close to the quantile value.
    E.g., for q=0.9, approximately 90% of targets should be below predictions.

    Args:
        predictions: Predicted quantile values
        targets: Ground truth values
        quantile: Quantile value

    Returns:
        Empirical coverage (proportion of targets <= predictions)
    """
    coverage = (targets <= predictions).float().mean().item()
    return coverage
