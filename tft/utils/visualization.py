"""
Visualization utilities for Temporal Fusion Transformer.

This module provides functions for plotting predictions, attention weights,
and variable importance for model interpretability.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Optional, List, Tuple, Dict
import pandas as pd


def plot_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    timestamps: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 6),
    title: str = "Predictions vs Actuals",
    save_path: Optional[str] = None,
):
    """
    Plot predictions with quantile intervals against actual values.

    Args:
        predictions: Predicted quantiles of shape (seq_len, num_quantiles)
        targets: Actual values of shape (seq_len,)
        quantiles: List of quantile values
        timestamps: Optional timestamps for x-axis
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)

    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Create x-axis
    if timestamps is None:
        timestamps = np.arange(len(targets))

    # Plot actual values
    plt.plot(timestamps, targets, 'k-', label='Actual', linewidth=2)

    # Plot median prediction
    if 0.5 in quantiles:
        median_idx = quantiles.index(0.5)
        plt.plot(
            timestamps,
            predictions[:, median_idx],
            'b-',
            label='Predicted (median)',
            linewidth=2,
        )

    # Plot prediction intervals
    if len(quantiles) >= 2:
        lower_idx = 0
        upper_idx = -1
        plt.fill_between(
            timestamps,
            predictions[:, lower_idx],
            predictions[:, upper_idx],
            alpha=0.3,
            label=f'Prediction interval ({quantiles[lower_idx]:.0%}-{quantiles[upper_idx]:.0%})',
        )

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_attention_weights(
    attention_weights: torch.Tensor,
    timestamps: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
):
    """
    Plot attention weight heatmap.

    Args:
        attention_weights: Attention weights of shape (num_heads, tgt_len, src_len)
                          or (tgt_len, src_len) for averaged attention
        timestamps: Optional timestamps for axes labels
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()

    # Average over heads if multiple heads
    if attention_weights.ndim == 3:
        num_heads = attention_weights.shape[0]
        fig, axes = plt.subplots(1, num_heads, figsize=(figsize[0] * num_heads / 4, figsize[1]))

        if num_heads == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            sns.heatmap(
                attention_weights[i],
                cmap='viridis',
                ax=ax,
                cbar=True,
                xticklabels=timestamps if timestamps is not None else 'auto',
                yticklabels=timestamps if timestamps is not None else 'auto',
            )
            ax.set_title(f'Head {i + 1}')
            ax.set_xlabel('Source Position')
            ax.set_ylabel('Target Position')

        plt.suptitle(title)
    else:
        plt.figure(figsize=figsize)
        sns.heatmap(
            attention_weights,
            cmap='viridis',
            cbar=True,
            xticklabels=timestamps if timestamps is not None else 'auto',
            yticklabels=timestamps if timestamps is not None else 'auto',
        )
        plt.xlabel('Source Position')
        plt.ylabel('Target Position')
        plt.title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_variable_importance(
    variable_weights: torch.Tensor,
    variable_names: List[str],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Variable Importance",
    save_path: Optional[str] = None,
):
    """
    Plot variable selection weights as a bar chart.

    Args:
        variable_weights: Variable selection weights of shape (num_variables,)
                         or (time_steps, num_variables) for temporal
        variable_names: Names of variables
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
    """
    if isinstance(variable_weights, torch.Tensor):
        variable_weights = variable_weights.cpu().numpy()

    # Average over time if temporal
    if variable_weights.ndim == 2:
        variable_weights = variable_weights.mean(axis=0)

    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Variable': variable_names,
        'Importance': variable_weights,
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=figsize)
    plt.barh(df['Variable'], df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Variable')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):
    """
    Plot training history (loss curves).

    Args:
        history: Dictionary with 'train_loss' and optionally 'val_loss'
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)

    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_quantile_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float],
    sample_idx: int = 0,
    figsize: Tuple[int, int] = (15, 6),
    save_path: Optional[str] = None,
):
    """
    Plot all quantile predictions for a single sample.

    Args:
        predictions: Predicted quantiles (batch_size, seq_len, num_quantiles)
        targets: Actual values (batch_size, seq_len)
        quantiles: List of quantile values
        sample_idx: Which sample to plot
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Extract single sample
    pred_sample = predictions[sample_idx]
    target_sample = targets[sample_idx]

    plt.figure(figsize=figsize)

    # Plot actual
    plt.plot(target_sample, 'k-', label='Actual', linewidth=2, alpha=0.7)

    # Plot each quantile
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles)))
    for i, (q, color) in enumerate(zip(quantiles, colors)):
        plt.plot(pred_sample[:, i], label=f'Q{q:.2f}', color=color, linewidth=1.5, alpha=0.8)

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'Quantile Predictions - Sample {sample_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_forecast_fan(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    historical_values: Optional[torch.Tensor] = None,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    figsize: Tuple[int, int] = (15, 6),
    title: str = "Forecast Fan Chart",
    save_path: Optional[str] = None,
):
    """
    Create a fan chart showing historical data and forecast with uncertainty.

    Args:
        predictions: Predicted quantiles
        targets: Actual future values
        historical_values: Historical values to show before forecast
        quantiles: List of quantile values
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if historical_values is not None and isinstance(historical_values, torch.Tensor):
        historical_values = historical_values.cpu().numpy()

    plt.figure(figsize=figsize)

    # Determine time indices
    forecast_len = len(targets)
    if historical_values is not None:
        hist_len = len(historical_values)
        hist_time = np.arange(hist_len)
        forecast_time = np.arange(hist_len, hist_len + forecast_len)

        # Plot historical
        plt.plot(hist_time, historical_values, 'k-', label='Historical', linewidth=2)
    else:
        forecast_time = np.arange(forecast_len)

    # Plot actual future
    plt.plot(forecast_time, targets, 'k-', label='Actual', linewidth=2)

    # Plot median forecast
    if 0.5 in quantiles:
        median_idx = quantiles.index(0.5)
        plt.plot(
            forecast_time,
            predictions[:, median_idx],
            'b--',
            label='Forecast (median)',
            linewidth=2,
        )

    # Plot prediction intervals (fan)
    colors = ['#d1e5f0', '#92c5de', '#4393c3']
    for i in range(len(quantiles) // 2):
        lower_idx = i
        upper_idx = len(quantiles) - 1 - i
        alpha = 0.3 - (i * 0.1)
        plt.fill_between(
            forecast_time,
            predictions[:, lower_idx],
            predictions[:, upper_idx],
            alpha=alpha,
            color=colors[min(i, len(colors) - 1)],
            label=f'{quantiles[lower_idx]:.0%}-{quantiles[upper_idx]:.0%} interval' if i == 0 else None,
        )

    # Add vertical line at forecast start
    if historical_values is not None:
        plt.axvline(x=hist_len, color='gray', linestyle='--', alpha=0.5)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
