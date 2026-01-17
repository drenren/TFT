"""
Evaluation metrics for Temporal Fusion Transformer.

This module provides metrics for evaluating probabilistic forecasting performance,
including quantile-specific metrics and traditional error metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile: float,
) -> float:
    """
    Compute quantile loss for a specific quantile.

    Args:
        predictions: Predicted values for the quantile
        targets: Ground truth values
        quantile: Quantile value (0 < q < 1)

    Returns:
        Quantile loss value
    """
    errors = targets - predictions
    loss = torch.max(
        (quantile - 1) * errors,
        quantile * errors,
    )
    return loss.mean().item()


def compute_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values

    Returns:
        MAE value
    """
    return torch.abs(predictions - targets).mean().item()


def compute_rmse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values

    Returns:
        RMSE value
    """
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def compute_mape(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute Mean Absolute Percentage Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value (in percentage)
    """
    percentage_errors = torch.abs((targets - predictions) / (targets + epsilon))
    return (percentage_errors.mean() * 100).item()


def compute_smape(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        epsilon: Small value to avoid division by zero

    Returns:
        SMAPE value (in percentage)
    """
    numerator = torch.abs(predictions - targets)
    denominator = (torch.abs(targets) + torch.abs(predictions)) / 2 + epsilon
    return (100 * (numerator / denominator).mean()).item()


def compute_r2_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute R² (coefficient of determination) score.

    Args:
        predictions: Predicted values
        targets: Ground truth values

    Returns:
        R² score
    """
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return r2.item()


def compute_quantile_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float],
) -> Dict[str, float]:
    """
    Compute metrics for all quantiles.

    Args:
        predictions: Predicted quantiles of shape (..., num_quantiles)
        targets: Ground truth values
        quantiles: List of quantile values

    Returns:
        Dictionary with metrics for each quantile
    """
    metrics = {}

    for i, q in enumerate(quantiles):
        q_pred = predictions[..., i]
        q_loss = compute_quantile_loss(q_pred, targets, q)
        metrics[f'q{int(q*100)}_loss'] = q_loss

        # Coverage (what proportion of targets are below this quantile)
        coverage = (targets <= q_pred).float().mean().item()
        metrics[f'q{int(q*100)}_coverage'] = coverage

    # Additional metrics for median (q=0.5)
    if 0.5 in quantiles:
        median_idx = quantiles.index(0.5)
        median_pred = predictions[..., median_idx]

        metrics['mae'] = compute_mae(median_pred, targets)
        metrics['rmse'] = compute_rmse(median_pred, targets)
        metrics['mape'] = compute_mape(median_pred, targets)
        metrics['smape'] = compute_smape(median_pred, targets)
        metrics['r2'] = compute_r2_score(median_pred, targets)

    return metrics


def compute_normalized_quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float],
    scale: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute normalized quantile losses.

    Args:
        predictions: Predicted quantiles
        targets: Ground truth values
        quantiles: List of quantile values
        scale: Optional scaling factor

    Returns:
        Dictionary with normalized losses
    """
    if scale is None:
        scale = torch.abs(targets).mean() + 1e-8

    metrics = {}

    for i, q in enumerate(quantiles):
        q_pred = predictions[..., i]
        q_loss = compute_quantile_loss(q_pred, targets, q)
        normalized_loss = q_loss / scale.item()
        metrics[f'normalized_q{int(q*100)}_loss'] = normalized_loss

    return metrics


def compute_prediction_intervals(
    predictions: torch.Tensor,
    quantiles: List[float],
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract prediction intervals from quantile predictions.

    Args:
        predictions: Predicted quantiles of shape (..., num_quantiles)
        quantiles: List of quantile values
        lower_quantile: Lower bound quantile
        upper_quantile: Upper bound quantile

    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    lower_idx = quantiles.index(lower_quantile)
    upper_idx = quantiles.index(upper_quantile)

    lower_bounds = predictions[..., lower_idx]
    upper_bounds = predictions[..., upper_idx]

    return lower_bounds, upper_bounds


def compute_interval_coverage(
    targets: torch.Tensor,
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
) -> float:
    """
    Compute empirical coverage of prediction intervals.

    Args:
        targets: Ground truth values
        lower_bounds: Lower bounds of prediction interval
        upper_bounds: Upper bounds of prediction interval

    Returns:
        Coverage proportion
    """
    in_interval = ((targets >= lower_bounds) & (targets <= upper_bounds)).float()
    return in_interval.mean().item()


def compute_interval_width(
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
) -> float:
    """
    Compute mean width of prediction intervals.

    Args:
        lower_bounds: Lower bounds
        upper_bounds: Upper bounds

    Returns:
        Mean interval width
    """
    widths = upper_bounds - lower_bounds
    return widths.mean().item()


class MetricsCalculator:
    """
    Helper class for computing and tracking metrics during training.

    Args:
        quantiles: List of quantiles
        device: Device for computations
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        device: str = 'cpu',
    ):
        self.quantiles = quantiles
        self.device = device

        # Accumulators
        self.reset()

    def reset(self):
        """Reset accumulators."""
        self.total_predictions = []
        self.total_targets = []
        self.num_samples = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Update accumulators with new batch.

        Args:
            predictions: Batch predictions
            targets: Batch targets
        """
        self.total_predictions.append(predictions.detach().cpu())
        self.total_targets.append(targets.detach().cpu())
        self.num_samples += targets.shape[0]

    def compute(self) -> Dict[str, float]:
        """
        Compute metrics over all accumulated data.

        Returns:
            Dictionary of metrics
        """
        if len(self.total_predictions) == 0:
            return {}

        # Concatenate all batches
        predictions = torch.cat(self.total_predictions, dim=0)
        targets = torch.cat(self.total_targets, dim=0)

        # Compute quantile metrics
        metrics = compute_quantile_metrics(predictions, targets, self.quantiles)

        # Compute normalized metrics
        normalized_metrics = compute_normalized_quantile_loss(
            predictions, targets, self.quantiles
        )
        metrics.update(normalized_metrics)

        # Compute prediction interval metrics
        if len(self.quantiles) >= 2:
            lower_bounds, upper_bounds = compute_prediction_intervals(
                predictions,
                self.quantiles,
                lower_quantile=self.quantiles[0],
                upper_quantile=self.quantiles[-1],
            )

            coverage = compute_interval_coverage(targets, lower_bounds, upper_bounds)
            width = compute_interval_width(lower_bounds, upper_bounds)

            metrics['interval_coverage'] = coverage
            metrics['interval_width'] = width

        return metrics

    def compute_and_reset(self) -> Dict[str, float]:
        """
        Compute metrics and reset accumulators.

        Returns:
            Dictionary of metrics
        """
        metrics = self.compute()
        self.reset()
        return metrics
