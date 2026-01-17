"""
Configuration management for Temporal Fusion Transformer.

This module provides dataclasses for managing hyperparameters and
configurations for the TFT model.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import yaml
import json


@dataclass
class TFTConfig:
    """
    Configuration for Temporal Fusion Transformer model.

    This dataclass holds all hyperparameters and settings for the TFT model,
    including input specifications, architecture parameters, and training settings.

    Input Specifications:
        static_variables: List of time-invariant feature names
        known_future: List of features known in advance (at forecast time)
        observed_only: List of features only available historically
        target: Name of the target variable to forecast

    Sequence Configuration:
        encoder_length: Length of historical sequence
        decoder_length: Length of forecast horizon

    Architecture Parameters:
        hidden_size: Hidden dimension for GRNs and other layers
        num_heads: Number of attention heads
        num_lstm_layers: Number of LSTM layers
        dropout: Dropout rate
        quantiles: List of quantiles for probabilistic forecasting

    Training Parameters:
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        max_epochs: Maximum number of training epochs
        gradient_clip_val: Maximum gradient norm for clipping
    """

    # Input feature specifications
    static_variables: List[str] = field(default_factory=list)
    known_future: List[str] = field(default_factory=list)
    observed_only: List[str] = field(default_factory=list)
    target: str = "target"

    # Sequence configuration
    encoder_length: int = 168  # Historical sequence length
    decoder_length: int = 24   # Forecast horizon

    # Embedding dimensions
    static_embedding_dim: int = 8
    time_varying_embedding_dim: int = 8

    # Architecture parameters
    hidden_size: int = 160
    num_heads: int = 4
    num_lstm_layers: int = 2
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Quantiles for probabilistic forecasting
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    weight_decay: float = 0.0

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Learning rate scheduler
    use_lr_scheduler: bool = True
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5

    # Device
    device: str = "auto"  # "auto", "cuda", "mps", or "cpu"

    @property
    def num_static_variables(self) -> int:
        """Number of static variables."""
        return len(self.static_variables)

    @property
    def num_known_future(self) -> int:
        """Number of known future variables."""
        return len(self.known_future)

    @property
    def num_observed_only(self) -> int:
        """Number of observed-only variables."""
        return len(self.observed_only)

    @property
    def num_encoder_variables(self) -> int:
        """Number of variables in encoder (observed + known)."""
        return self.num_known_future + self.num_observed_only

    @property
    def num_decoder_variables(self) -> int:
        """Number of variables in decoder (known only)."""
        return self.num_known_future

    @property
    def all_variables(self) -> List[str]:
        """All variable names."""
        return self.static_variables + self.known_future + self.observed_only

    @property
    def num_quantiles(self) -> int:
        """Number of quantiles for forecasting."""
        return len(self.quantiles)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self, filepath: str) -> None:
        """Save config to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_json(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TFTConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, filepath: str) -> 'TFTConfig':
        """Load config from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, filepath: str) -> 'TFTConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.encoder_length > 0, "encoder_length must be positive"
        assert self.decoder_length > 0, "decoder_length must be positive"
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert all(0 < q < 1 for q in self.quantiles), \
            "quantiles must be in (0, 1)"
        assert self.quantiles == sorted(self.quantiles), \
            "quantiles must be sorted in ascending order"
        assert self.target is not None and len(self.target) > 0, \
            "target variable must be specified"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.gradient_clip_val > 0, "gradient_clip_val must be positive"

    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()

    def __repr__(self) -> str:
        """String representation of config."""
        lines = ["TFTConfig("]
        lines.append(f"  Input Variables:")
        lines.append(f"    Static: {self.static_variables}")
        lines.append(f"    Known Future: {self.known_future}")
        lines.append(f"    Observed Only: {self.observed_only}")
        lines.append(f"    Target: {self.target}")
        lines.append(f"  Sequence:")
        lines.append(f"    Encoder Length: {self.encoder_length}")
        lines.append(f"    Decoder Length: {self.decoder_length}")
        lines.append(f"  Architecture:")
        lines.append(f"    Hidden Size: {self.hidden_size}")
        lines.append(f"    Attention Heads: {self.num_heads}")
        lines.append(f"    LSTM Layers: {self.num_lstm_layers}")
        lines.append(f"    Dropout: {self.dropout}")
        lines.append(f"  Quantiles: {self.quantiles}")
        lines.append(f"  Training:")
        lines.append(f"    Batch Size: {self.batch_size}")
        lines.append(f"    Learning Rate: {self.learning_rate}")
        lines.append(f"    Max Epochs: {self.max_epochs}")
        lines.append(")")
        return "\n".join(lines)
