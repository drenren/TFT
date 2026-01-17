"""
Core building blocks for the Temporal Fusion Transformer.

This module implements the fundamental components used throughout the TFT architecture:
- Gated Linear Unit (GLU): Gating mechanism for controlling information flow
- Gated Residual Network (GRN): Core building block with skip connections
- GateAddNorm: Combines gating, skip connections, and layer normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) activation function.

    GLU splits the input in half along the feature dimension,
    applies sigmoid to one half (gate) and element-wise multiplies
    with the other half (value).

    GLU(x) = σ(W_g x + b_g) ⊙ (W_v x + b_v)

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension (if None, defaults to input_dim)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

        # Linear layer that outputs 2x the desired dimension
        # (one for the value, one for the gate)
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GLU.

        Args:
            x: Input tensor of shape (batch_size, ..., input_dim)

        Returns:
            Output tensor of shape (batch_size, ..., output_dim)
        """
        x = self.fc(x)
        x = self.dropout(x)

        # Split into value and gate
        value, gate = torch.chunk(x, 2, dim=-1)

        # Apply sigmoid to gate and multiply with value
        return value * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - Core building block of TFT.

    The GRN applies non-linear processing with gating and skip connections.
    It optionally accepts a context vector for conditioning.

    Architecture:
        1. Optional context projection
        2. Linear layer 1 (input -> hidden)
        3. ELU activation
        4. Linear layer 2 (hidden -> hidden)
        5. GLU gating
        6. Dropout
        7. Layer normalization + skip connection

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension (if None, defaults to input_dim)
        context_dim: Optional context vector dimension
        dropout: Dropout rate
        use_time_distributed: Whether to apply same weights across time steps
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_time_distributed: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.context_dim = context_dim
        self.use_time_distributed = use_time_distributed

        # Layer 1: Project input to hidden dimension
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Context projection (if context is provided)
        if context_dim is not None:
            self.context_projection = nn.Linear(context_dim, hidden_dim, bias=False)

        # Layer 2: Project hidden to hidden dimension
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # GLU for gating
        self.glu = GatedLinearUnit(hidden_dim, self.output_dim, dropout=dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_dim)

        # Skip connection (project if dimensions don't match)
        if input_dim != self.output_dim:
            self.skip_projection = nn.Linear(input_dim, self.output_dim)
        else:
            self.skip_projection = None

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GRN.

        Args:
            x: Input tensor of shape (batch_size, ..., input_dim)
            context: Optional context tensor of shape (batch_size, context_dim)

        Returns:
            Output tensor of shape (batch_size, ..., output_dim)
        """
        # Save for skip connection
        residual = x

        # Layer 1
        x = self.fc1(x)

        # Add context if provided
        if context is not None and self.context_dim is not None:
            context_proj = self.context_projection(context)
            # Broadcast context if needed (for time-distributed application)
            if x.dim() > context_proj.dim():
                context_proj = context_proj.unsqueeze(1)
            x = x + context_proj

        # ELU activation
        x = F.elu(x)

        # Layer 2
        x = self.fc2(x)
        x = F.elu(x)

        # GLU gating
        x = self.glu(x)
        x = self.dropout(x)

        # Skip connection
        if self.skip_projection is not None:
            residual = self.skip_projection(residual)

        # Add & Norm
        x = self.layer_norm(x + residual)

        return x


class GateAddNorm(nn.Module):
    """
    Gating with Add & Norm (Gated Skip Connection).

    This module implements a gated skip connection followed by layer normalization.
    It's used to combine the output of a sub-layer with its input.

    Args:
        input_dim: Input feature dimension
        dropout: Dropout rate
    """

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.glu = GatedLinearUnit(input_dim, input_dim, dropout=dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply gating, skip connection, and layer normalization.

        Args:
            x: Input tensor (output from sub-layer)
            residual: Residual tensor (input to sub-layer)

        Returns:
            Normalized output tensor
        """
        # Apply GLU to input
        x = self.glu(x)
        x = self.dropout(x)

        # Add residual and normalize
        x = self.layer_norm(x + residual)

        return x


class TimeDistributed(nn.Module):
    """
    Wrapper to apply a module across the time dimension.

    This is useful for applying the same operation to each time step
    in a sequence independently.

    Args:
        module: PyTorch module to apply across time
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply module across time dimension.

        Args:
            x: Input tensor of shape (batch_size, time_steps, ...)

        Returns:
            Output tensor with same time structure
        """
        if x.dim() <= 2:
            return self.module(x)

        # Reshape to (batch_size * time_steps, ...)
        batch_size, time_steps = x.shape[0], x.shape[1]
        x_reshaped = x.reshape(batch_size * time_steps, *x.shape[2:])

        # Apply module
        y = self.module(x_reshaped)

        # Reshape back to (batch_size, time_steps, ...)
        y = y.reshape(batch_size, time_steps, *y.shape[1:])

        return y


def apply_gating_layer(
    x: torch.Tensor,
    hidden_dim: int,
    dropout: float = 0.0,
    activation: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Apply a gating layer with optional activation.

    Helper function for applying GLU-based gating.

    Args:
        x: Input tensor
        hidden_dim: Hidden dimension for GLU
        dropout: Dropout rate
        activation: Optional activation function to apply before GLU

    Returns:
        Gated output tensor
    """
    if activation is not None:
        x = activation(x)

    glu = GatedLinearUnit(x.shape[-1], hidden_dim, dropout=dropout)
    return glu(x)


def get_linear_layer(
    input_dim: int,
    output_dim: int,
    use_time_distributed: bool = False,
) -> nn.Module:
    """
    Create a linear layer, optionally wrapped in TimeDistributed.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        use_time_distributed: Whether to wrap in TimeDistributed

    Returns:
        Linear layer module
    """
    linear = nn.Linear(input_dim, output_dim)

    if use_time_distributed:
        return TimeDistributed(linear)

    return linear
