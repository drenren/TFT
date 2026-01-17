"""
Temporal Fusion Transformer (TFT) for PyTorch.

A complete implementation of the Temporal Fusion Transformer for
interpretable multi-horizon time series forecasting.

Example usage:
    >>> from tft import TemporalFusionTransformer, TFTConfig, TFTTrainer
    >>> from tft.data import create_dataloaders
    >>>
    >>> # Create configuration
    >>> config = TFTConfig(
    ...     static_variables=['store_id'],
    ...     known_future=['day_of_week', 'hour'],
    ...     observed_only=['sales'],
    ...     target='sales',
    ...     encoder_length=168,
    ...     decoder_length=24,
    ... )
    >>>
    >>> # Create model
    >>> model = TemporalFusionTransformer(config)
    >>>
    >>> # Prepare data
    >>> train_loader, val_loader, _ = create_dataloaders(
    ...     train_data, val_data, None, config
    ... )
    >>>
    >>> # Train
    >>> trainer = TFTTrainer(model, config)
    >>> trainer.fit(train_loader, val_loader, epochs=100)
"""

from .models import TemporalFusionTransformer
from .utils import TFTConfig
from .training import TFTTrainer, QuantileLoss
from .data import create_dataloaders, TimeSeriesDataset

__version__ = '0.1.0'

__all__ = [
    'TemporalFusionTransformer',
    'TFTConfig',
    'TFTTrainer',
    'QuantileLoss',
    'create_dataloaders',
    'TimeSeriesDataset',
]
