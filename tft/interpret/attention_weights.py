"""
Attention weight analysis and visualization for TFT.

This module provides tools for extracting and analyzing attention weights
from the Temporal Fusion Transformer for interpretability.
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def extract_attention_weights(
    model_outputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Extract attention weights from model outputs.

    Args:
        model_outputs: Dictionary from model forward pass

    Returns:
        Attention weights tensor of shape (batch_size, num_heads, tgt_len, src_len)
    """
    if 'attention_weights' not in model_outputs:
        raise ValueError("Model outputs do not contain attention weights")

    return model_outputs['attention_weights']


def average_attention_over_heads(
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Average attention weights across all heads.

    Args:
        attention_weights: Tensor of shape (batch_size, num_heads, tgt_len, src_len)

    Returns:
        Averaged weights of shape (batch_size, tgt_len, src_len)
    """
    return attention_weights.mean(dim=1)


def get_temporal_attention_patterns(
    attention_weights: torch.Tensor,
    sample_idx: int = 0,
) -> np.ndarray:
    """
    Extract temporal attention patterns for a specific sample.

    Args:
        attention_weights: Attention weights
        sample_idx: Which sample to extract

    Returns:
        Attention pattern for the sample
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()

    return attention_weights[sample_idx]


def analyze_attention_focus(
    attention_weights: torch.Tensor,
    top_k: int = 5,
) -> Dict[str, List]:
    """
    Analyze which positions receive the most attention.

    Args:
        attention_weights: Attention weights (batch_size, tgt_len, src_len)
        top_k: Number of top positions to report

    Returns:
        Dictionary with analysis results
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()

    # Average over batch and target positions
    avg_attention = attention_weights.mean(axis=(0, 1))

    # Get top-k positions
    top_indices = np.argsort(avg_attention)[-top_k:][::-1]
    top_weights = avg_attention[top_indices]

    return {
        'top_positions': top_indices.tolist(),
        'top_weights': top_weights.tolist(),
        'mean_attention': avg_attention.tolist(),
    }


def compute_attention_entropy(
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute entropy of attention distributions.

    Higher entropy indicates more uniform attention (less focused).
    Lower entropy indicates more concentrated attention (more focused).

    Args:
        attention_weights: Attention weights

    Returns:
        Entropy values
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    entropy = -(attention_weights * torch.log(attention_weights + eps)).sum(dim=-1)
    return entropy


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    sample_idx: int = 0,
    head_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot attention weight heatmap.

    Args:
        attention_weights: Attention weights (batch_size, num_heads, tgt_len, src_len)
        sample_idx: Which sample to plot
        head_idx: Which head to plot (if None, averages over all heads)
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()

    # Extract sample
    sample_weights = attention_weights[sample_idx]

    if head_idx is not None:
        # Plot specific head
        weights_to_plot = sample_weights[head_idx]
        if title is None:
            title = f'Attention Weights - Sample {sample_idx}, Head {head_idx}'
    else:
        # Average over heads
        weights_to_plot = sample_weights.mean(axis=0)
        if title is None:
            title = f'Attention Weights (Averaged) - Sample {sample_idx}'

    plt.figure(figsize=figsize)
    sns.heatmap(
        weights_to_plot,
        cmap='viridis',
        cbar=True,
        xticklabels=5,
        yticklabels=5,
    )
    plt.xlabel('Source Position')
    plt.ylabel('Target Position')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_attention_by_head(
    attention_weights: torch.Tensor,
    sample_idx: int = 0,
    figsize: Tuple[int, int] = (16, 4),
    save_path: Optional[str] = None,
):
    """
    Plot attention weights for all heads side by side.

    Args:
        attention_weights: Attention weights (batch_size, num_heads, tgt_len, src_len)
        sample_idx: Which sample to plot
        figsize: Figure size per head
        save_path: Optional path to save figure
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()

    sample_weights = attention_weights[sample_idx]
    num_heads = sample_weights.shape[0]

    fig, axes = plt.subplots(1, num_heads, figsize=(figsize[0], figsize[1]))

    if num_heads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        sns.heatmap(
            sample_weights[i],
            cmap='viridis',
            ax=ax,
            cbar=True,
            xticklabels=5,
            yticklabels=5,
        )
        ax.set_title(f'Head {i}')
        ax.set_xlabel('Source')
        ax.set_ylabel('Target')

    plt.suptitle(f'Attention Weights by Head - Sample {sample_idx}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def compute_attention_rollout(
    attention_weights: List[torch.Tensor],
) -> torch.Tensor:
    """
    Compute attention rollout across multiple layers.

    This is useful for multi-layer attention models to understand
    the cumulative attention flow.

    Args:
        attention_weights: List of attention weight tensors from each layer

    Returns:
        Rolled-out attention weights
    """
    # Start with identity matrix
    result = torch.eye(attention_weights[0].shape[-1])

    # Multiply attention matrices
    for weights in attention_weights:
        # Average over heads and batch
        avg_weights = weights.mean(dim=(0, 1))
        result = torch.matmul(avg_weights, result)

    return result
