"""
Interpretable Multi-Head Attention for Temporal Fusion Transformer.

This module implements the attention mechanism used in TFT, which uses
additive attention (not scaled dot-product) for better interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention mechanism.

    Unlike standard scaled dot-product attention, this uses additive attention
    which provides better interpretability. Each head learns different aspects
    of temporal relationships.

    The attention mechanism computes:
        Attention(Q, K, V) = softmax(V_a tanh(W_q Q + W_k K)) V

    where V_a, W_q, W_k are learned parameters.

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Query, Key, Value projections for all heads (parallelized)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Additive attention score projection
        # Each head has its own attention score function
        self.v_a = nn.Parameter(torch.randn(num_heads, self.d_head))

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, tgt_len, d_model)
            key: Key tensor of shape (batch_size, src_len, d_model)
            value: Value tensor of shape (batch_size, src_len, d_model)
            mask: Optional mask tensor of shape (batch_size, tgt_len, src_len)
                  or (batch_size, 1, src_len). True values indicate positions
                  that should be masked (not attended to).

        Returns:
            Tuple of:
                - output: Attended output (batch_size, tgt_len, d_model)
                - attention_weights: Attention weights (batch_size, num_heads, tgt_len, src_len)
        """
        batch_size = query.shape[0]
        tgt_len = query.shape[1]
        src_len = key.shape[1]

        # Project Q, K, V
        Q = self.w_q(query)  # (batch_size, tgt_len, d_model)
        K = self.w_k(key)    # (batch_size, src_len, d_model)
        V = self.w_v(value)  # (batch_size, src_len, d_model)

        # Split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_head)
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.d_head).transpose(1, 2)

        # Compute additive attention scores
        # Q: (batch_size, num_heads, tgt_len, d_head)
        # K: (batch_size, num_heads, src_len, d_head)

        # Expand for broadcasting
        # Q: (batch_size, num_heads, tgt_len, 1, d_head)
        # K: (batch_size, num_heads, 1, src_len, d_head)
        Q_expanded = Q.unsqueeze(3)
        K_expanded = K.unsqueeze(2)

        # Additive attention: tanh(Q + K)
        # (batch_size, num_heads, tgt_len, src_len, d_head)
        combined = torch.tanh(Q_expanded + K_expanded)

        # Project to scalar scores using v_a
        # v_a: (num_heads, d_head)
        # Reshape v_a for broadcasting: (1, num_heads, 1, 1, d_head)
        v_a_expanded = self.v_a.view(1, self.num_heads, 1, 1, self.d_head)

        # Compute attention scores: (batch_size, num_heads, tgt_len, src_len)
        attention_scores = (combined * v_a_expanded).sum(dim=-1) * self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention scores shape
            if mask.dim() == 3:
                # (batch_size, tgt_len, src_len) -> (batch_size, 1, tgt_len, src_len)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                # (batch_size, src_len) -> (batch_size, 1, 1, src_len)
                mask = mask.unsqueeze(1).unsqueeze(2)

            # Fill masked positions with large negative value
            attention_scores = attention_scores.masked_fill(mask, -1e9)

        # Apply softmax to get attention weights
        # (batch_size, num_heads, tgt_len, src_len)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # attention_weights: (batch_size, num_heads, tgt_len, src_len)
        # V: (batch_size, num_heads, src_len, d_head)
        # output: (batch_size, num_heads, tgt_len, d_head)
        attended = torch.matmul(attention_weights, V)

        # Concatenate heads
        # (batch_size, num_heads, tgt_len, d_head) -> (batch_size, tgt_len, d_model)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)

        # Final output projection
        output = self.w_o(attended)
        output = self.dropout(output)

        return output, attention_weights


class TemporalSelfAttention(nn.Module):
    """
    Temporal Self-Attention layer with position-wise feed-forward network.

    This wraps the InterpretableMultiHeadAttention and adds a feed-forward
    network for additional non-linear processing.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = InterpretableMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            Tuple of:
                - output: Attended output (batch_size, seq_len, d_model)
                - attention_weights: Attention weights for interpretability
        """
        # Save residual
        residual = x

        # Self-attention (Q = K = V = x)
        attended, attention_weights = self.attention(
            query=x,
            key=x,
            value=x,
            mask=mask,
        )

        # Add & Norm
        x = self.layer_norm(residual + self.dropout(attended))

        return x, attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models (optional for TFT).

    Adds sinusoidal positional encodings to the input embeddings.

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
