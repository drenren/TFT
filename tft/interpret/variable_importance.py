"""
Variable importance analysis for TFT.

This module provides tools for analyzing variable selection weights
to understand which features are most important for predictions.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def extract_variable_selection_weights(
    model_outputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Extract variable selection weights from model outputs.

    Args:
        model_outputs: Dictionary from model forward pass

    Returns:
        Dictionary with encoder and decoder variable selection weights
    """
    weights = {}

    if 'encoder_variable_selection' in model_outputs:
        weights['encoder'] = model_outputs['encoder_variable_selection']

    if 'decoder_variable_selection' in model_outputs:
        weights['decoder'] = model_outputs['decoder_variable_selection']

    return weights


def compute_average_variable_importance(
    variable_weights: torch.Tensor,
) -> np.ndarray:
    """
    Compute average importance for each variable.

    Args:
        variable_weights: Weights of shape (batch_size, time_steps, num_variables)
                         or (batch_size, num_variables)

    Returns:
        Average importance per variable
    """
    if isinstance(variable_weights, torch.Tensor):
        variable_weights = variable_weights.cpu().numpy()

    # Average over batch and time dimensions
    if variable_weights.ndim == 3:
        avg_importance = variable_weights.mean(axis=(0, 1))
    elif variable_weights.ndim == 2:
        avg_importance = variable_weights.mean(axis=0)
    else:
        raise ValueError(f"Unexpected weight dimensions: {variable_weights.shape}")

    return avg_importance


def rank_variables_by_importance(
    variable_weights: torch.Tensor,
    variable_names: List[str],
) -> pd.DataFrame:
    """
    Rank variables by their average importance.

    Args:
        variable_weights: Variable selection weights
        variable_names: Names of variables

    Returns:
        DataFrame with variables ranked by importance
    """
    avg_importance = compute_average_variable_importance(variable_weights)

    df = pd.DataFrame({
        'variable': variable_names,
        'importance': avg_importance,
    })

    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    return df


def analyze_temporal_variable_importance(
    variable_weights: torch.Tensor,
    variable_names: List[str],
) -> pd.DataFrame:
    """
    Analyze how variable importance changes over time.

    Args:
        variable_weights: Weights of shape (batch_size, time_steps, num_variables)
        variable_names: Names of variables

    Returns:
        DataFrame with temporal importance patterns
    """
    if isinstance(variable_weights, torch.Tensor):
        variable_weights = variable_weights.cpu().numpy()

    if variable_weights.ndim != 3:
        raise ValueError("Expected 3D weights (batch, time, variables)")

    # Average over batch
    avg_weights = variable_weights.mean(axis=0)  # (time_steps, num_variables)

    # Create DataFrame
    df = pd.DataFrame(avg_weights, columns=variable_names)
    df['time_step'] = df.index

    return df


def compute_variable_interactions(
    encoder_weights: torch.Tensor,
    decoder_weights: torch.Tensor,
    encoder_vars: List[str],
    decoder_vars: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Analyze interactions between encoder and decoder variables.

    Args:
        encoder_weights: Encoder variable selection weights
        decoder_weights: Decoder variable selection weights
        encoder_vars: Encoder variable names
        decoder_vars: Decoder variable names

    Returns:
        Dictionary of interactions
    """
    enc_importance = compute_average_variable_importance(encoder_weights)
    dec_importance = compute_average_variable_importance(decoder_weights)

    interactions = {
        'encoder': {name: float(imp) for name, imp in zip(encoder_vars, enc_importance)},
        'decoder': {name: float(imp) for name, imp in zip(decoder_vars, dec_importance)},
    }

    # Find common variables
    common_vars = set(encoder_vars) & set(decoder_vars)
    if common_vars:
        interactions['common'] = {}
        for var in common_vars:
            enc_idx = encoder_vars.index(var)
            dec_idx = decoder_vars.index(var)
            interactions['common'][var] = {
                'encoder_importance': float(enc_importance[enc_idx]),
                'decoder_importance': float(dec_importance[dec_idx]),
            }

    return interactions


def plot_variable_importance(
    variable_weights: torch.Tensor,
    variable_names: List[str],
    top_k: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Variable Importance",
    save_path: Optional[str] = None,
):
    """
    Plot variable importance as a bar chart.

    Args:
        variable_weights: Variable selection weights
        variable_names: Names of variables
        top_k: Number of top variables to show (if None, shows all)
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
    """
    df = rank_variables_by_importance(variable_weights, variable_names)

    if top_k is not None:
        df = df.head(top_k)

    plt.figure(figsize=figsize)
    plt.barh(df['variable'][::-1], df['importance'][::-1])
    plt.xlabel('Importance')
    plt.ylabel('Variable')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_temporal_variable_importance(
    variable_weights: torch.Tensor,
    variable_names: List[str],
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Variable Importance Over Time",
    save_path: Optional[str] = None,
):
    """
    Plot how variable importance changes over time.

    Args:
        variable_weights: Weights (batch_size, time_steps, num_variables)
        variable_names: Names of variables
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
    """
    df = analyze_temporal_variable_importance(variable_weights, variable_names)

    plt.figure(figsize=figsize)

    # Plot each variable
    for var in variable_names:
        plt.plot(df['time_step'], df[var], label=var, linewidth=2, marker='o', markersize=4)

    plt.xlabel('Time Step')
    plt.ylabel('Importance')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_variable_importance_heatmap(
    variable_weights: torch.Tensor,
    variable_names: List[str],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Variable Importance Heatmap",
    save_path: Optional[str] = None,
):
    """
    Plot variable importance as a heatmap over time.

    Args:
        variable_weights: Weights (batch_size, time_steps, num_variables)
        variable_names: Names of variables
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
    """
    df = analyze_temporal_variable_importance(variable_weights, variable_names)

    # Drop time_step column for heatmap
    df_heatmap = df.drop('time_step', axis=1)

    plt.figure(figsize=figsize)
    sns.heatmap(
        df_heatmap.T,
        cmap='YlOrRd',
        cbar=True,
        xticklabels=5,
        yticklabels=variable_names,
    )
    plt.xlabel('Time Step')
    plt.ylabel('Variable')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def compare_encoder_decoder_importance(
    encoder_weights: torch.Tensor,
    decoder_weights: torch.Tensor,
    encoder_vars: List[str],
    decoder_vars: List[str],
    common_vars_only: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
):
    """
    Compare variable importance between encoder and decoder.

    Args:
        encoder_weights: Encoder variable selection weights
        decoder_weights: Decoder variable selection weights
        encoder_vars: Encoder variable names
        decoder_vars: Decoder variable names
        common_vars_only: Whether to show only common variables
        figsize: Figure size
        save_path: Optional path to save figure
    """
    enc_importance = compute_average_variable_importance(encoder_weights)
    dec_importance = compute_average_variable_importance(decoder_weights)

    if common_vars_only:
        common_vars = list(set(encoder_vars) & set(decoder_vars))
        enc_idx = [encoder_vars.index(v) for v in common_vars]
        dec_idx = [decoder_vars.index(v) for v in common_vars]

        enc_vals = enc_importance[enc_idx]
        dec_vals = dec_importance[dec_idx]
        vars_to_plot = common_vars
    else:
        # Show all variables (may not align)
        enc_vals = enc_importance
        dec_vals = np.zeros(len(encoder_vars))
        vars_to_plot = encoder_vars

    x = np.arange(len(vars_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, enc_vals, width, label='Encoder', alpha=0.8)
    ax.bar(x + width/2, dec_vals, width, label='Decoder', alpha=0.8)

    ax.set_xlabel('Variable')
    ax.set_ylabel('Importance')
    ax.set_title('Encoder vs Decoder Variable Importance')
    ax.set_xticks(x)
    ax.set_xticklabels(vars_to_plot, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
