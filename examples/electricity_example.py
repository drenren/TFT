"""
Electricity Load Diagrams example for Temporal Fusion Transformer.

This script demonstrates training TFT on the ElectricityLoadDiagrams20112014 dataset
from the UCI Machine Learning Repository. The dataset contains electricity consumption
of 370 clients recorded at 15-minute intervals from 2011 to 2014.

Dataset: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

Workflow:
1. Download and preprocess the electricity dataset
2. Create time-based features
3. Prepare data for TFT
4. Train and evaluate the model
"""

import sys
sys.path.append('..')

import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tft.models import TemporalFusionTransformer
from tft.data import create_dataloaders
from tft.training import TFTTrainer, EarlyStopping, ModelCheckpoint
from tft.utils import TFTConfig, get_device, print_device_info
from tft.utils.visualization import plot_predictions, plot_training_history


# Dataset URLs
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "LD2011_2014.txt"


def download_electricity_data(data_dir: Path = DATA_DIR) -> Path:
    """
    Download the ElectricityLoadDiagrams20112014 dataset.

    Args:
        data_dir: Directory to save the data

    Returns:
        Path to the downloaded data file
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "LD2011_2014.txt.zip"
    txt_path = data_dir / "LD2011_2014.txt"

    if txt_path.exists():
        print(f"   Data already exists at {txt_path}")
        return txt_path

    print(f"   Downloading from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, zip_path)

    print(f"   Extracting to {data_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Clean up zip file
    zip_path.unlink()

    return txt_path


def load_electricity_data(
    data_path: Path,
    num_clients: int = 10,
    resample_freq: str = "1h",
) -> pd.DataFrame:
    """
    Load and preprocess the electricity dataset.

    The raw data has 15-minute intervals. We resample to hourly for faster training.

    Args:
        data_path: Path to the data file
        num_clients: Number of clients to use (for faster training, use subset)
        resample_freq: Resampling frequency (e.g., "1h" for hourly)

    Returns:
        Preprocessed DataFrame in long format
    """
    print(f"   Loading data from {data_path}...")

    # Read the data (semicolon separated, decimal comma)
    df = pd.read_csv(
        data_path,
        sep=';',
        decimal=',',
        index_col=0,
        parse_dates=True,
    )

    # The first row might have column names, drop if necessary
    if df.index[0] == df.index[1]:
        df = df.iloc[1:]

    # Convert index to datetime if not already
    df.index = pd.to_datetime(df.index)

    # Select subset of clients for faster training
    selected_clients = df.columns[:num_clients].tolist()
    df = df[selected_clients]

    print(f"   Selected {num_clients} clients: {selected_clients[:5]}...")

    # Resample to specified frequency (sum consumption within each period)
    df = df.resample(resample_freq).sum()

    # Remove any rows with all zeros or NaN
    df = df.dropna()
    df = df[(df != 0).any(axis=1)]

    print(f"   Resampled to {resample_freq}, shape: {df.shape}")

    # Convert to long format
    df = df.reset_index()
    df = df.melt(
        id_vars=['index'],
        var_name='client_id',
        value_name='consumption',
    )
    df = df.rename(columns={'index': 'datetime'})

    # Create client_id as integer for embedding
    client_mapping = {c: i for i, c in enumerate(selected_clients)}
    df['client_id_num'] = df['client_id'].map(client_mapping)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features for the TFT model.

    Args:
        df: DataFrame with datetime column

    Returns:
        DataFrame with added time features
    """
    df = df.copy()

    # Extract time components
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)

    # Cyclical encoding for periodic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Create time index (continuous integer index per client)
    df = df.sort_values(['client_id', 'datetime'])
    df['time_idx'] = df.groupby('client_id').cumcount()

    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(float)

    return df


def prepare_data(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """
    Split data into train, validation, and test sets by time.

    Args:
        df: Full dataset
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Get unique time indices
    unique_times = sorted(df['time_idx'].unique())
    n_times = len(unique_times)

    train_end = int(n_times * train_frac)
    val_end = int(n_times * (train_frac + val_frac))

    train_times = set(unique_times[:train_end])
    val_times = set(unique_times[train_end:val_end])
    test_times = set(unique_times[val_end:])

    train_df = df[df['time_idx'].isin(train_times)].copy()
    val_df = df[df['time_idx'].isin(val_times)].copy()
    test_df = df[df['time_idx'].isin(test_times)].copy()

    return train_df, val_df, test_df


def main():
    """Main execution function."""

    print("=" * 80)
    print("Temporal Fusion Transformer - Electricity Load Diagrams Example")
    print("=" * 80)

    # Configuration - reduced for faster CPU training
    # Increase these values for better results with GPU
    NUM_CLIENTS = 3       # Use small subset for demo (full dataset has 370)
    RESAMPLE_FREQ = "4h"  # Resample to 4-hourly for faster training
    ENCODER_LENGTH = 42   # ~1 week of 4-hourly data (42 * 4 = 168 hours)
    DECODER_LENGTH = 6    # Predict 24 hours ahead (6 * 4 = 24 hours)

    # 1. Download data
    print("\n1. Downloading electricity dataset...")
    data_path = download_electricity_data()

    # 2. Load and preprocess data
    print("\n2. Loading and preprocessing data...")
    df = load_electricity_data(
        data_path,
        num_clients=NUM_CLIENTS,
        resample_freq=RESAMPLE_FREQ,
    )
    print(f"   Total samples: {len(df)}")

    # 3. Add time features
    print("\n3. Adding time features...")
    df = add_time_features(df)
    print(f"   Features: {list(df.columns)}")

    # 4. Prepare data splits
    print("\n4. Preparing data splits...")
    train_df, val_df, test_df = prepare_data(df)
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")

    # 5. Create TFT configuration
    print("\n5. Creating TFT configuration...")
    config = TFTConfig(
        # Input features
        static_variables=['client_id_num'],
        known_future=[
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'is_weekend',
        ],
        observed_only=[],  # Only the target is observed
        target='consumption',

        # Sequence lengths
        encoder_length=ENCODER_LENGTH,
        decoder_length=DECODER_LENGTH,

        # Architecture - reduced for faster CPU training
        hidden_size=64,
        num_heads=4,
        num_lstm_layers=1,
        dropout=0.1,

        # Quantiles for probabilistic forecasting
        quantiles=[0.1, 0.5, 0.9],

        # Training parameters
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=3,  # Reduced for demo
        gradient_clip_val=1.0,
    )

    print(f"   Config created:")
    print(f"   - Encoder length: {config.encoder_length} steps (~1 week at 4h intervals)")
    print(f"   - Decoder length: {config.decoder_length} steps (~24 hours at 4h intervals)")
    print(f"   - Hidden size: {config.hidden_size}")
    print(f"   - Attention heads: {config.num_heads}")

    # 6. Create data loaders
    print("\n6. Creating data loaders...")
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

    # 7. Create model
    print("\n7. Creating TFT model...")
    print_device_info()
    device = get_device("auto")

    model = TemporalFusionTransformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {num_params:,} parameters")

    # 8. Create trainer
    print("\n8. Setting up trainer...")
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
            filepath='electricity_best_model.pth',
            monitor='val_loss',
            save_best_only=True,
            verbose=True,
        ),
    ]

    # 9. Train model
    print("\n9. Training model...")
    print("-" * 80)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.max_epochs,
        callbacks=callbacks,
    )
    print("-" * 80)

    # 10. Evaluate on test set
    print("\n10. Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    print("   Test metrics:")
    for metric, value in test_metrics.items():
        print(f"   - {metric}: {value:.6f}")

    # 11. Generate predictions
    print("\n11. Generating predictions...")
    results = trainer.predict(
        test_loader,
        return_attention=True,
        return_variable_selection=True,
    )

    predictions = results['predictions']
    targets = results['targets']
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Targets shape: {targets.shape}")

    # 12. Visualize results
    print("\n12. Visualizing results...")

    # Plot training history
    plot_training_history(
        trainer.history,
        save_path='electricity_training_history.png',
    )

    # Plot predictions for first sample
    sample_idx = 0
    plot_predictions(
        predictions=predictions[sample_idx],
        targets=targets[sample_idx],
        quantiles=config.quantiles,
        title="TFT Electricity Consumption Forecast - Sample 0",
        save_path='electricity_predictions_sample_0.png',
    )

    # 13. Analyze variable importance (if available)
    if 'variable_selection_weights' in results:
        print("\n13. Variable Selection Weights:")
        var_weights = results['variable_selection_weights']
        if 'encoder' in var_weights:
            print("   Encoder variable weights:")
            encoder_weights = var_weights['encoder'].mean(dim=0)
            known_vars = config.known_future
            for i, var in enumerate(known_vars):
                if i < len(encoder_weights):
                    print(f"   - {var}: {encoder_weights[i].item():.4f}")

    print("\n" + "=" * 80)
    print("Electricity example completed successfully!")
    print("=" * 80)
    print("\nSummary:")
    print(f"- Dataset: ElectricityLoadDiagrams20112014")
    print(f"- Clients used: {NUM_CLIENTS}")
    print(f"- Resampling: {RESAMPLE_FREQ}")
    print(f"- Final train loss: {trainer.history['train_loss'][-1]:.6f}")
    print(f"- Final val loss:   {trainer.history['val_loss'][-1]:.6f}")
    print(f"- Test loss:        {test_metrics['loss']:.6f}")
    print(f"- Best validation loss: {trainer.best_val_loss:.6f}")
    print("\nPlots saved:")
    print("- electricity_training_history.png")
    print("- electricity_predictions_sample_0.png")
    print("\nModel checkpoint saved: electricity_best_model.pth")


if __name__ == '__main__':
    main()
