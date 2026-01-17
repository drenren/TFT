"""
Temporal Fusion Transformer main model.

This module implements the complete TFT architecture, integrating all components
including variable selection, LSTMs, attention, and quantile forecasting.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import sys
sys.path.append('..')

from .components import GatedResidualNetwork
from .variable_selection import (
    StaticCovariateEncoder,
    TemporalVariableSelection,
)
from .attention import TemporalSelfAttention


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for interpretable multi-horizon forecasting.

    The TFT integrates several key components:
    1. Static covariate encoders for time-invariant features
    2. Variable selection networks for sparse feature selection
    3. LSTM encoders/decoders for temporal processing
    4. Multi-head attention for learning temporal relationships
    5. Quantile output heads for probabilistic forecasting

    Args:
        config: TFTConfig object with model hyperparameters
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_quantiles = config.num_quantiles
        self.encoder_length = config.encoder_length
        self.decoder_length = config.decoder_length

        # Static covariate encoder
        if config.num_static_variables > 0:
            self.static_encoder = StaticCovariateEncoder(
                num_static_variables=config.num_static_variables,
                static_input_dim=config.static_embedding_dim,
                hidden_dim=config.hidden_size,
                dropout=config.dropout,
            )
        else:
            self.static_encoder = None

        # Historical/Encoder variable selection
        if config.num_encoder_variables > 0:
            self.encoder_vsn = TemporalVariableSelection(
                input_dim=config.time_varying_embedding_dim,
                num_inputs=config.num_encoder_variables,
                hidden_dim=config.hidden_size,
                output_dim=config.hidden_size,
                context_dim=config.hidden_size if config.num_static_variables > 0 else None,
                dropout=config.dropout,
            )

        # Future/Decoder variable selection
        if config.num_decoder_variables > 0:
            self.decoder_vsn = TemporalVariableSelection(
                input_dim=config.time_varying_embedding_dim,
                num_inputs=config.num_decoder_variables,
                hidden_dim=config.hidden_size,
                output_dim=config.hidden_size,
                context_dim=config.hidden_size if config.num_static_variables > 0 else None,
                dropout=config.dropout,
            )

        # LSTM Encoder (for historical data)
        self.lstm_encoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        # LSTM Decoder (for future known inputs)
        self.lstm_decoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        # Gate to apply after LSTM
        self.post_lstm_gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GLU(dim=-1) if hasattr(nn, 'GLU') else nn.Identity(),
        )

        # Static enrichment layer (add static context to temporal features)
        self.static_enrichment = GatedResidualNetwork(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            output_dim=config.hidden_size,
            context_dim=config.hidden_size if config.num_static_variables > 0 else None,
            dropout=config.dropout,
        )

        # Multi-head attention
        self.attention = TemporalSelfAttention(
            d_model=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
        )

        # Post-attention GRN (position-wise feed-forward)
        self.post_attention_grn = GatedResidualNetwork(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            output_dim=config.hidden_size,
            dropout=config.dropout,
        )

        # Quantile output layers
        # Shared layer for all quantiles
        self.output_layer = nn.Linear(config.hidden_size, config.hidden_size)

        # Separate output head for each quantile
        self.quantile_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, 1)
            for _ in range(config.num_quantiles)
        ])

    def init_hidden_state(
        self,
        batch_size: int,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden and cell states.

        If static context is provided, use it to initialize the states.
        Otherwise, initialize with zeros.

        Args:
            batch_size: Batch size
            context: Optional context tensor from static encoder

        Returns:
            Tuple of (hidden_state, cell_state), each of shape
            (num_layers, batch_size, hidden_size)
        """
        device = next(self.parameters()).device
        num_layers = self.config.num_lstm_layers

        if context is not None:
            # Use context to initialize hidden state
            # Expand context to all LSTM layers
            hidden = context.unsqueeze(0).repeat(num_layers, 1, 1)
            cell = torch.zeros_like(hidden)
        else:
            # Initialize with zeros
            hidden = torch.zeros(
                num_layers, batch_size, self.hidden_size,
                device=device,
            )
            cell = torch.zeros(
                num_layers, batch_size, self.hidden_size,
                device=device,
            )

        return hidden, cell

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TFT.

        Args:
            batch: Dictionary containing:
                - 'static_features': (batch_size, num_static, embedding_dim) if static vars exist
                - 'encoder_features': (batch_size, encoder_len, num_encoder, embedding_dim)
                - 'decoder_features': (batch_size, decoder_len, num_decoder, embedding_dim)

        Returns:
            Dictionary containing:
                - 'predictions': (batch_size, decoder_len, num_quantiles) - Quantile predictions
                - 'attention_weights': (batch_size, num_heads, decoder_len, encoder_len+decoder_len)
                - 'encoder_variable_selection': (batch_size, encoder_len, num_encoder)
                - 'decoder_variable_selection': (batch_size, decoder_len, num_decoder)
        """
        batch_size = batch['encoder_features'].shape[0] if 'encoder_features' in batch else batch['decoder_features'].shape[0]

        # 1. Process static covariates (if any)
        if self.static_encoder is not None and 'static_features' in batch:
            (static_embedding,
             context_selection,
             context_enrichment,
             context_lstm,
             context_state) = self.static_encoder(batch['static_features'])
        else:
            device = next(self.parameters()).device
            static_embedding = torch.zeros(batch_size, self.hidden_size, device=device)
            context_selection = None
            context_enrichment = None
            context_lstm = None
            context_state = None

        # 2. Historical/Encoder processing
        if 'encoder_features' in batch and self.config.num_encoder_variables > 0:
            # Variable selection for encoder
            encoder_output, encoder_variable_selection = self.encoder_vsn(
                batch['encoder_features'],
                context=context_selection,
            )
            # encoder_output: (batch_size, encoder_len, hidden_size)

            # LSTM encoder
            init_hidden, init_cell = self.init_hidden_state(batch_size, context_lstm)
            encoder_lstm_out, (encoder_hidden, encoder_cell) = self.lstm_encoder(
                encoder_output,
                (init_hidden, init_cell),
            )
        else:
            encoder_lstm_out = None
            encoder_variable_selection = None
            encoder_hidden, encoder_cell = self.init_hidden_state(batch_size, context_lstm)

        # 3. Future/Decoder processing
        if 'decoder_features' in batch and self.config.num_decoder_variables > 0:
            # Variable selection for decoder
            decoder_output, decoder_variable_selection = self.decoder_vsn(
                batch['decoder_features'],
                context=context_selection,
            )
            # decoder_output: (batch_size, decoder_len, hidden_size)

            # LSTM decoder (continues from encoder state)
            decoder_lstm_out, _ = self.lstm_decoder(
                decoder_output,
                (encoder_hidden, encoder_cell),
            )
        else:
            decoder_lstm_out = None
            decoder_variable_selection = None

        # 4. Concatenate encoder and decoder LSTM outputs
        if encoder_lstm_out is not None and decoder_lstm_out is not None:
            lstm_output = torch.cat([encoder_lstm_out, decoder_lstm_out], dim=1)
        elif encoder_lstm_out is not None:
            lstm_output = encoder_lstm_out
        elif decoder_lstm_out is not None:
            lstm_output = decoder_lstm_out
        else:
            raise ValueError("Either encoder or decoder features must be provided")

        # 5. Gate the LSTM output
        # Apply gating to control information flow
        gated_lstm = self.post_lstm_gate(lstm_output)

        # 6. Static enrichment
        # Add static context to temporal features
        enriched = self.static_enrichment(
            gated_lstm if isinstance(self.post_lstm_gate[-1], nn.Identity) else lstm_output,
            context=context_enrichment,
        )

        # 7. Multi-head attention
        # Apply self-attention across entire sequence
        attended, attention_weights = self.attention(enriched)

        # 8. Position-wise feed-forward (GRN)
        temporal_features = self.post_attention_grn(attended)

        # 9. Extract decoder outputs (only forecast horizon)
        # temporal_features: (batch_size, encoder_len + decoder_len, hidden_size)
        # We only need the last decoder_len positions for forecasting
        forecast_features = temporal_features[:, -self.decoder_length:, :]

        # 10. Generate quantile predictions
        # Shared transformation
        output_features = torch.relu(self.output_layer(forecast_features))

        # Separate head for each quantile
        quantile_predictions = []
        for quantile_head in self.quantile_heads:
            quantile_pred = quantile_head(output_features).squeeze(-1)
            quantile_predictions.append(quantile_pred)

        # Stack quantiles: (batch_size, decoder_len, num_quantiles)
        predictions = torch.stack(quantile_predictions, dim=-1)

        # 11. Prepare output dictionary
        output = {
            'predictions': predictions,
            'attention_weights': attention_weights,
        }

        if encoder_variable_selection is not None:
            output['encoder_variable_selection'] = encoder_variable_selection

        if decoder_variable_selection is not None:
            output['decoder_variable_selection'] = decoder_variable_selection

        return output

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions (same as forward, but in eval mode).

        Args:
            batch: Input batch dictionary

        Returns:
            Output dictionary with predictions and interpretability info
        """
        self.eval()
        with torch.no_grad():
            return self.forward(batch)
