"""
Synthetic data example for Temporal Fusion Transformer.

This script demonstrates the complete workflow:
1. Generate synthetic time series data
2. Prepare data for TFT
3. Create and train the model
4. Evaluate and visualize results
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import torch

from tft.models import TemporalFusionTransformer
from tft.data import create_dataloaders
from tft.training import TFTTrainer, EarlyStopping, ModelCheckpoint
from tft.utils import TFTConfig
from tft.utils.visualization import plot_predictions, plot_training_history


def generate_synthetic_data(
    num_samples: int = 1000,
    num_series: int = 5,
    noise_level: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic time series data.

    Creates multiple sinusoidal time series with:
    - Trend component
    - Seasonal component
    - Noise
    - Known future features (time-based)
    - Static features (series ID)

    Args:
        num_samples: Number of time steps per series
        num_series: Number of different time series
        noise_level: Standard deviation of noise
        seed: Random seed

    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(seed)

    data_list = []

    for series_id in range(num_series):
        # Time index
        time_idx = np.arange(num_samples)

        # Components
        trend = 0.02 * time_idx
        seasonality = 10 * np.sin(2 * np.pi * time_idx / 50)  # Period of 50
        noise = noise_level * np.random.randn(num_samples)

        # Target value
        target = trend + seasonality + noise + series_id * 2

        # Time features (known in advance)
        hour = (time_idx % 24).astype(float)
        day_of_week = ((time_idx // 24) % 7).astype(float)

        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Additional observed feature (correlated with target)
        temperature = 20 + 5 * np.sin(2 * np.pi * time_idx / 50) + \
                     np.random.randn(num_samples) * 0.5

        # Create DataFrame for this series
        series_df = pd.DataFrame({
            'series_id': series_id,
            'time_idx': time_idx,
            'target': target,
            'hour': hour,
            'day_of_week': day_of_week,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'temperature': temperature,
        })

        data_list.append(series_df)

    # Combine all series
    df = pd.concat(data_list, axis=0, ignore_index=True)

    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Split data into train, validation, and test sets.

    Args:
        df: Full dataset

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Split by time (70% train, 15% val, 15% test)
    unique_times = sorted(df['time_idx'].unique())
    n_times = len(unique_times)

    train_end = int(n_times * 0.7)
    val_end = int(n_times * 0.85)

    train_times = unique_times[:train_end]
    val_times = unique_times[train_end:val_end]
    test_times = unique_times[val_end:]

    train_df = df[df['time_idx'].isin(train_times)].copy()
    val_df = df[df['time_idx'].isin(val_times)].copy()
    test_df = df[df['time_idx'].isin(test_times)].copy()

    return train_df, val_df, test_df


def main():
    """Main execution function."""

    print("=" * 80)
    print("Temporal Fusion Transformer - Synthetic Data Example")
    print("=" * 80)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_data(
        num_samples=1000,
        num_series=5,
        noise_level=0.5,
        seed=42,
    )
    print(f"   Generated {len(df)} samples across {df['series_id'].nunique()} series")
    print(f"   Columns: {list(df.columns)}")

    # 2. Prepare data splits
    print("\n2. Preparing data splits...")
    train_df, val_df, test_df = prepare_data(df)
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")

    # 3. Create TFT configuration
    print("\n3. Creating TFT configuration...")
    config = TFTConfig(
        # Input features
        static_variables=['series_id'],
        known_future=['hour_sin', 'hour_cos', 'day_of_week'],
        observed_only=['temperature'],
        target='target',

        # Sequence lengths
        encoder_length=50,   # Look back 50 steps
        decoder_length=10,   # Forecast 10 steps ahead

        # Architecture
        hidden_size=64,
        num_heads=4,
        num_lstm_layers=1,
        dropout=0.1,

        # Quantiles
        quantiles=[0.1, 0.5, 0.9],

        # Training
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=3,  # Reduced for quick demo
        gradient_clip_val=1.0,
    )
    print(f"   Config created:")
    print(f"   - Encoder length: {config.encoder_length}")
    print(f"   - Decoder length: {config.decoder_length}")
    print(f"   - Hidden size: {config.hidden_size}")

    # 4. Create data loaders
    print("\n4. Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        config=config,
        batch_size=config.batch_size,
        num_workers=0,
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")

    # 5. Create model
    print("\n5. Creating TFT model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")

    model = TemporalFusionTransformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {num_params:,} parameters")

    # 6. Create trainer
    print("\n6. Setting up trainer...")
    trainer = TFTTrainer(
        model=model,
        config=config,
        device=device,
    )

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=1e-4,
            verbose=True,
        ),
        ModelCheckpoint(
            filepath='best_model.pth',
            monitor='val_loss',
            save_best_only=True,
            verbose=True,
        ),
    ]

    # 7. Train model
    print("\n7. Training model...")
    print("-" * 80)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.max_epochs,
        callbacks=callbacks,
    )
    print("-" * 80)

    # 8. Evaluate on test set
    print("\n8. Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    print("   Test metrics:")
    for metric, value in test_metrics.items():
        print(f"   - {metric}: {value:.6f}")

    # 9. Generate predictions
    print("\n9. Generating predictions...")
    results = trainer.predict(
        test_loader,
        return_attention=True,
        return_variable_selection=True,
    )

    predictions = results['predictions']
    targets = results['targets']
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Targets shape: {targets.shape}")

    # 10. Visualize results
    print("\n10. Visualizing results...")

    # Plot training history
    plot_training_history(
        trainer.history,
        save_path='training_history.png',
    )

    # Plot predictions for first sample
    sample_idx = 0
    plot_predictions(
        predictions=predictions[sample_idx],
        targets=targets[sample_idx],
        quantiles=config.quantiles,
        title="TFT Predictions - Sample 0",
        save_path='predictions_sample_0.png',
    )

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nSummary:")
    print(f"- Final train loss: {trainer.history['train_loss'][-1]:.6f}")
    print(f"- Final val loss:   {trainer.history['val_loss'][-1]:.6f}")
    print(f"- Test loss:        {test_metrics['loss']:.6f}")
    print(f"- Best validation loss: {trainer.best_val_loss:.6f}")
    print("\nPlots saved:")
    print("- training_history.png")
    print("- predictions_sample_0.png")
    print("\nModel checkpoint saved: best_model.pth")


if __name__ == '__main__':
    main()
