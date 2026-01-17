"""
Variable Selection Network for Temporal Fusion Transformer.

The Variable Selection Network (VSN) is a key component of TFT that provides
interpretable feature selection through learned sparse weights.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from .components import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network for sparse feature selection.

    The VSN processes each input variable independently through GRNs,
    computes softmax-based selection weights, and outputs a weighted
    combination of the processed features along with the selection weights
    for interpretability.

    Architecture:
        For each variable:
            1. Flatten and concatenate all variables
            2. Pass through GRN to compute selection weights
            3. Apply softmax for sparse selection
            4. Process each variable independently through GRNs
            5. Weight and combine processed variables

    Args:
        input_dim: Dimension of each input variable
        num_inputs: Number of input variables
        hidden_dim: Hidden dimension for GRNs
        output_dim: Output dimension
        context_dim: Optional context vector dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        # GRN for computing variable selection weights
        # Input: flattened concatenation of all variables
        self.flattening_grn = GatedResidualNetwork(
            input_dim=input_dim * num_inputs,
            hidden_dim=hidden_dim,
            output_dim=num_inputs,  # One weight per variable
            context_dim=context_dim,
            dropout=dropout,
        )

        # Separate GRN for each variable
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                context_dim=None,  # Context already used in flattening GRN
                dropout=dropout,
            )
            for _ in range(num_inputs)
        ])

        # Final GRN to process the weighted combination
        self.output_grn = GatedResidualNetwork(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            context_dim=None,
            dropout=dropout,
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        variables: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Variable Selection Network.

        Args:
            variables: Input tensor of shape (batch_size, num_inputs, input_dim)
                      or (batch_size, time_steps, num_inputs, input_dim)
            context: Optional context tensor of shape (batch_size, context_dim)

        Returns:
            Tuple of:
                - selected_features: Weighted combination of processed variables
                  Shape: (batch_size, output_dim) or (batch_size, time_steps, output_dim)
                - selection_weights: Variable selection weights for interpretability
                  Shape: (batch_size, num_inputs) or (batch_size, time_steps, num_inputs)
        """
        # Handle both 3D and 4D inputs (with or without time dimension)
        time_distributed = variables.dim() == 4

        if time_distributed:
            batch_size, time_steps, num_inputs, input_dim = variables.shape
            # Reshape to (batch_size * time_steps, num_inputs, input_dim)
            variables = variables.reshape(batch_size * time_steps, num_inputs, input_dim)
            # Also repeat context for each time step if provided
            if context is not None:
                context = context.unsqueeze(1).repeat(1, time_steps, 1).reshape(batch_size * time_steps, -1)
        else:
            batch_size, num_inputs, input_dim = variables.shape

        # Flatten all variables: (batch_size, num_inputs * input_dim)
        flattened = variables.reshape(variables.shape[0], -1)

        # Compute variable selection weights via GRN
        # Output shape: (batch_size, num_inputs)
        selection_logits = self.flattening_grn(flattened, context=context)
        selection_weights = self.softmax(selection_logits)

        # Process each variable independently through its GRN
        processed_variables = []
        for i, grn in enumerate(self.variable_grns):
            # Extract i-th variable: (batch_size, input_dim)
            var = variables[:, i, :]
            # Process through GRN: (batch_size, output_dim)
            processed = grn(var)
            processed_variables.append(processed)

        # Stack processed variables: (batch_size, num_inputs, output_dim)
        processed_variables = torch.stack(processed_variables, dim=1)

        # Apply selection weights
        # Expand weights: (batch_size, num_inputs, 1)
        weights_expanded = selection_weights.unsqueeze(-1)

        # Weighted sum: (batch_size, output_dim)
        selected = torch.sum(processed_variables * weights_expanded, dim=1)

        # Apply final GRN
        output = self.output_grn(selected)

        # Reshape back if time-distributed
        if time_distributed:
            output = output.reshape(batch_size, time_steps, -1)
            selection_weights = selection_weights.reshape(batch_size, time_steps, -1)

        return output, selection_weights


class StaticCovariateEncoder(nn.Module):
    """
    Encoder for static (time-invariant) covariates.

    Processes static features and generates multiple context vectors
    for conditioning different parts of the TFT model.

    Args:
        num_static_variables: Number of static input variables
        static_input_dim: Dimension of each static variable
        hidden_dim: Hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_static_variables: int,
        static_input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_static_variables = num_static_variables
        self.static_input_dim = static_input_dim
        self.hidden_dim = hidden_dim

        if num_static_variables > 0:
            # Variable selection for static variables
            self.static_vsn = VariableSelectionNetwork(
                input_dim=static_input_dim,
                num_inputs=num_static_variables,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                context_dim=None,  # No context for static encoder
                dropout=dropout,
            )

            # GRNs for generating different context vectors
            # c_s: Static context for variable selection in temporal encoders
            self.context_grn_selection = GatedResidualNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )

            # c_c: Context for enrichment of temporal features
            self.context_grn_enrichment = GatedResidualNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )

            # c_h: Context for LSTM initialization
            self.context_grn_lstm = GatedResidualNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )

            # c_e: Context for static enrichment
            self.context_grn_state = GatedResidualNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )

    def forward(
        self,
        static_variables: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode static covariates into context vectors.

        Args:
            static_variables: Static input tensor
                Shape: (batch_size, num_static_variables, static_input_dim)

        Returns:
            Tuple of context vectors:
                - static_embedding: Processed static features (batch_size, hidden_dim)
                - context_selection: c_s for VSN (batch_size, hidden_dim)
                - context_enrichment: c_c for enrichment (batch_size, hidden_dim)
                - context_lstm: c_h for LSTM init (batch_size, hidden_dim)
                - context_state: c_e for state (batch_size, hidden_dim)
        """
        if self.num_static_variables == 0:
            # Return zero contexts if no static variables
            batch_size = 1  # This shouldn't be called if no static vars
            device = next(self.parameters()).device
            zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
            return zeros, zeros, zeros, zeros, zeros

        # Process static variables through VSN
        static_embedding, static_weights = self.static_vsn(static_variables)

        # Generate different context vectors
        context_selection = self.context_grn_selection(static_embedding)
        context_enrichment = self.context_grn_enrichment(static_embedding)
        context_lstm = self.context_grn_lstm(static_embedding)
        context_state = self.context_grn_state(static_embedding)

        return (
            static_embedding,
            context_selection,
            context_enrichment,
            context_lstm,
            context_state,
        )


class TemporalVariableSelection(nn.Module):
    """
    Variable Selection Network for temporal (time-varying) inputs.

    This wraps the base VariableSelectionNetwork to handle temporal sequences.
    It applies the same VSN across all time steps.

    Args:
        input_dim: Dimension of each input variable
        num_inputs: Number of input variables
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        context_dim: Optional context vector dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vsn = VariableSelectionNetwork(
            input_dim=input_dim,
            num_inputs=num_inputs,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            context_dim=context_dim,
            dropout=dropout,
        )

    def forward(
        self,
        temporal_variables: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply VSN across temporal sequence.

        Args:
            temporal_variables: Input tensor
                Shape: (batch_size, time_steps, num_inputs, input_dim)
            context: Optional context tensor
                Shape: (batch_size, context_dim)

        Returns:
            Tuple of:
                - selected_features: (batch_size, time_steps, output_dim)
                - selection_weights: (batch_size, time_steps, num_inputs)
        """
        return self.vsn(temporal_variables, context=context)
