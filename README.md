# Temporal Fusion Transformer (TFT)

PyTorch implementation of the Temporal Fusion Transformer for interpretable multi-horizon time series forecasting.

## Overview

This repository implements the Temporal Fusion Transformer (TFT) architecture from the paper:

> **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting**
> Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister
> [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)

The TFT is a powerful attention-based architecture designed for multi-horizon forecasting with:
- **High performance**: State-of-the-art accuracy on various time series benchmarks
- **Interpretability**: Built-in attention weights and variable selection for understanding predictions
- **Flexibility**: Handles multiple types of inputs (static, known future, observed historical)
- **Probabilistic forecasting**: Produces quantile predictions for uncertainty estimation

## Key Features

### Model Architecture
- **Gated Residual Networks (GRN)**: Core building blocks with skip connections
- **Variable Selection Networks (VSN)**: Sparse feature selection with interpretable weights
- **Multi-head Attention**: Temporal relationships modeling
- **LSTM Encoders**: Separate processing for historical and future sequences
- **Quantile Outputs**: Probabilistic forecasting with multiple quantiles

### Implementation Features
- Clean, modular PyTorch implementation
- Comprehensive training utilities (trainer, callbacks, metrics)
- Data preprocessing and windowing for time series
- Interpretability tools (attention visualization, feature importance)
- Unit tests and examples

## Installation

### From source

```bash
git clone https://github.com/drenren/TFT.git
cd TFT
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- scikit-learn >= 1.3.0

See `requirements.txt` for complete dependencies.

### Hardware Acceleration

TFT automatically detects and uses the best available device:

- **CUDA (NVIDIA GPUs)**: Automatically used if available for fastest training
- **MPS (Apple Silicon)**: Automatically used on M1/M2/M3 Macs for GPU acceleration
- **CPU**: Fallback option, works on all systems

To check available devices:
```python
from tft.utils import print_device_info
print_device_info()
```

Or test device detection:
```bash
python examples/test_device.py
```

The device is automatically selected when creating a trainer:
```python
# Automatic device selection (recommended)
trainer = TFTTrainer(model, config)  # Uses best available device

# Manual device selection
trainer = TFTTrainer(model, config, device="cuda")  # Force CUDA
trainer = TFTTrainer(model, config, device="mps")   # Force MPS
trainer = TFTTrainer(model, config, device="cpu")   # Force CPU
```

## Quick Start

```python
from tft.models import TemporalFusionTransformer
from tft.data import TimeSeriesDataset, create_dataloaders
from tft.training import TFTTrainer, QuantileLoss
from tft.utils import TFTConfig

# Define configuration
config = TFTConfig(
    # Input features
    static_variables=['store_id', 'product_category'],
    known_future=['day_of_week', 'hour', 'price'],
    observed_only=['sales', 'temperature'],
    target='sales',

    # Sequence lengths
    encoder_length=168,  # 1 week of hourly data
    decoder_length=24,   # Forecast 24 hours ahead

    # Model architecture
    hidden_size=160,
    num_heads=4,
    num_lstm_layers=2,
    dropout=0.1,

    # Quantiles for probabilistic forecasting
    quantiles=[0.1, 0.5, 0.9],
)

# Create model
model = TemporalFusionTransformer(config)

# Prepare data
train_loader, val_loader = create_dataloaders(
    data=your_dataframe,
    config=config,
    batch_size=64,
)

# Train
trainer = TFTTrainer(model, config)
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
)

# Make predictions
predictions, attention_weights, variable_importance = model.predict(test_data)
```

## Project Structure

```
TFT/
├── tft/                          # Main package
│   ├── models/                   # Model architectures
│   │   ├── components.py         # Core building blocks (GRN, GLU)
│   │   ├── variable_selection.py # Variable Selection Networks
│   │   ├── attention.py          # Multi-head attention
│   │   └── temporal_fusion_transformer.py  # Main TFT model
│   ├── data/                     # Data handling
│   │   ├── dataset.py            # PyTorch Dataset
│   │   ├── preprocessing.py      # Preprocessing utilities
│   │   └── scalers.py            # Time series scalers
│   ├── training/                 # Training utilities
│   │   ├── trainer.py            # Training loop
│   │   ├── losses.py             # Quantile loss
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── callbacks.py          # Callbacks (early stopping, etc.)
│   ├── utils/                    # Utilities
│   │   ├── config.py             # Configuration management
│   │   └── visualization.py      # Plotting utilities
│   └── interpret/                # Interpretability
│       ├── attention_weights.py  # Attention visualization
│       └── variable_importance.py # Feature importance
├── examples/                     # Example scripts
│   └── synthetic_example.py      # Synthetic data training
├── tests/                        # Unit tests
└── configs/                      # Configuration files
```

## Model Architecture

The TFT processes three types of inputs:

1. **Static covariates**: Time-invariant features (e.g., store ID, product category)
2. **Known future inputs**: Features known in advance (e.g., day of week, promotions)
3. **Observed inputs**: Features only available historically (e.g., past sales, weather)

### Data Flow

```
Static Features → Variable Selection → Context Vectors
                                            ↓
Historical Data → Variable Selection → LSTM Encoder ─┐
                                                       ├→ Attention → GRN → Quantile Outputs
Future Known → Variable Selection → LSTM Decoder ────┘
```

### Key Components

**Gated Residual Network (GRN)**
- Core building block with gated linear units (GLU)
- Skip connections and layer normalization
- Optional context vector conditioning

**Variable Selection Network (VSN)**
- Sparse feature selection via softmax weights
- Per-variable processing through GRNs
- Interpretable feature importance scores

**Multi-head Attention**
- Additive attention mechanism (not scaled dot-product)
- Captures temporal dependencies
- Interpretable attention weights

**Quantile Forecasting**
- Multiple output heads for different quantiles
- Pinball loss for training
- Uncertainty estimation

## Usage Examples

### Synthetic Data Example

```bash
python examples/synthetic_example.py
```

This creates a synthetic sine wave dataset with noise and trains a TFT model.

### Custom Dataset

```python
import pandas as pd
from tft.data import TimeSeriesDataset

# Your time series dataframe
# Columns: timestamp, store_id, sales, temperature, day_of_week, ...
df = pd.read_csv('your_data.csv')

# Create dataset
dataset = TimeSeriesDataset(
    data=df,
    config=config,
    mode='train',
)

# Create DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

## Training

### Basic Training

```python
from tft.training import TFTTrainer

trainer = TFTTrainer(
    model=model,
    config=config,
    optimizer='adam',
    learning_rate=1e-3,
)

trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
)
```

### With Callbacks

```python
from tft.training.callbacks import EarlyStopping, ModelCheckpoint

trainer = TFTTrainer(model, config)
trainer.fit(
    train_loader,
    val_loader,
    epochs=100,
    callbacks=[
        EarlyStopping(patience=10, min_delta=1e-4),
        ModelCheckpoint(filepath='best_model.pth', save_best_only=True),
    ],
)
```

## Interpretability

### Attention Weights

```python
from tft.interpret import plot_attention_weights

# Get attention weights from model
predictions, attention_weights, _ = model.predict(test_data)

# Visualize temporal attention patterns
plot_attention_weights(attention_weights, timestamps=test_timestamps)
```

### Variable Importance

```python
from tft.interpret import plot_variable_importance

# Get variable selection weights
_, _, variable_importance = model.predict(test_data)

# Plot feature importance
plot_variable_importance(
    variable_importance,
    feature_names=config.all_variables,
)
```

## Performance

Expected performance on standard benchmarks (from original paper):

| Dataset | P50 Loss | P90 Loss |
|---------|----------|----------|
| Electricity | 0.051 | 0.025 |
| Traffic | 0.073 | 0.042 |
| Volatility | 0.154 | 0.089 |

## Testing

Run unit tests:

```bash
pytest tests/
```

Run specific test:

```bash
pytest tests/test_components.py
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{lim2021temporal,
  title={Temporal fusion transformers for interpretable multi-horizon time series forecasting},
  author={Lim, Bryan and Arik, Sercan O and Loeff, Nicolas and Pfister, Tomas},
  journal={International Journal of Forecasting},
  volume={37},
  number={4},
  pages={1748--1764},
  year={2021},
  publisher={Elsevier}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Original TFT paper by Lim et al.
- PyTorch team for the deep learning framework
