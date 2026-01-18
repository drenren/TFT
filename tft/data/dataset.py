"""
PyTorch Dataset for Temporal Fusion Transformer.

This module provides Dataset classes for creating time series windows
with proper handling of static, known future, and observed features.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .scalers import TimeSeriesScaler


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting with TFT.

    Creates windowed sequences with:
    - encoder_length time steps of historical data
    - decoder_length time steps of forecast horizon

    Handles three types of inputs:
    - Static covariates (time-invariant)
    - Known future inputs (known at forecast time)
    - Observed inputs (only available historically)

    Args:
        data: DataFrame with time series data
        config: TFTConfig with model configuration
        static_scalers: Optional dict of scalers for static features
        continuous_scalers: Optional dict of scalers for continuous features
        mode: 'train', 'val', or 'test'
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config,
        static_scalers: Optional[Dict] = None,
        continuous_scalers: Optional[Dict] = None,
        mode: str = 'train',
    ):
        self.data = data
        self.config = config
        self.mode = mode

        self.encoder_length = config.encoder_length
        self.decoder_length = config.decoder_length
        self.total_length = self.encoder_length + self.decoder_length

        # Feature lists
        self.static_variables = config.static_variables
        self.known_future = config.known_future
        self.observed_only = config.observed_only
        self.target = config.target

        # Scalers
        self.static_scalers = static_scalers or {}
        self.continuous_scalers = continuous_scalers or {}

        # Create valid indices (positions where we have enough data)
        self.valid_indices = self._create_valid_indices()

    def _create_valid_indices(self) -> List[int]:
        """
        Create list of valid starting indices for windows.

        An index is valid if we have encoder_length + decoder_length
        consecutive time steps available.
        """
        valid_indices = []
        n = len(self.data)

        for i in range(n - self.total_length + 1):
            valid_indices.append(i)

        return valid_indices

    def __len__(self) -> int:
        """Number of valid windows in the dataset."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single windowed sample.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
                - static_features: (num_static, embedding_dim)
                - encoder_features: (encoder_len, num_encoder, embedding_dim)
                - decoder_features: (decoder_len, num_decoder, embedding_dim)
                - target: (decoder_len,) - target values for forecast horizon
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.total_length

        # Get window of data
        window_data = self.data.iloc[start_idx:end_idx]

        batch = {}

        # 1. Static features (same across all time steps)
        if len(self.static_variables) > 0:
            static_features = []
            for var in self.static_variables:
                value = window_data[var].iloc[0]  # Take first value (constant)
                # Convert to tensor and repeat to match embedding_dim
                if isinstance(value, (int, float)):
                    feature = torch.tensor([value], dtype=torch.float32)
                else:
                    feature = torch.tensor([value], dtype=torch.float32).flatten()

                # Repeat to match static_embedding_dim
                feature = feature.repeat(self.config.static_embedding_dim)
                static_features.append(feature)

            # Stack: (num_static, static_embedding_dim)
            batch['static_features'] = torch.stack(static_features)

        # 2. Encoder (historical) features
        # Encoder uses: observed_only + known_future
        encoder_data = window_data.iloc[:self.encoder_length]

        if len(self.observed_only) + len(self.known_future) > 0:
            encoder_features = []

            # Add observed_only features
            for var in self.observed_only:
                values = torch.tensor(
                    encoder_data[var].values,
                    dtype=torch.float32,
                ).unsqueeze(-1)  # (encoder_len, 1)
                # Repeat to match time_varying_embedding_dim
                values = values.repeat(1, self.config.time_varying_embedding_dim)  # (encoder_len, embedding_dim)
                encoder_features.append(values)

            # Add known_future features
            for var in self.known_future:
                values = torch.tensor(
                    encoder_data[var].values,
                    dtype=torch.float32,
                ).unsqueeze(-1)  # (encoder_len, 1)
                # Repeat to match time_varying_embedding_dim
                values = values.repeat(1, self.config.time_varying_embedding_dim)  # (encoder_len, embedding_dim)
                encoder_features.append(values)

            # Stack: (encoder_len, num_encoder_vars, embedding_dim)
            encoder_features = torch.stack(encoder_features, dim=1)
            batch['encoder_features'] = encoder_features

        # 3. Decoder (future) features
        # Decoder uses only: known_future
        decoder_data = window_data.iloc[self.encoder_length:]

        if len(self.known_future) > 0:
            decoder_features = []

            for var in self.known_future:
                values = torch.tensor(
                    decoder_data[var].values,
                    dtype=torch.float32,
                ).unsqueeze(-1)  # (decoder_len, 1)
                # Repeat to match time_varying_embedding_dim
                values = values.repeat(1, self.config.time_varying_embedding_dim)  # (decoder_len, embedding_dim)
                decoder_features.append(values)

            # Stack: (decoder_len, num_decoder_vars, embedding_dim)
            decoder_features = torch.stack(decoder_features, dim=1)
            batch['decoder_features'] = decoder_features

        # 4. Target values (for loss computation)
        target_values = torch.tensor(
            decoder_data[self.target].values,
            dtype=torch.float32,
        )
        batch['target'] = target_values

        return batch


def create_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: Optional[pd.DataFrame],
    config,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for train, validation, and test sets.

    This function handles:
    - Creating scalers from training data
    - Normalizing all datasets with training statistics
    - Creating PyTorch DataLoaders with proper batching

    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        test_data: Test DataFrame (optional)
        config: TFTConfig object
        batch_size: Batch size (if None, uses config.batch_size)
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.batch_size

    # Create and fit scalers on training data
    continuous_vars = [v for v in config.all_variables if v != config.target]
    continuous_vars.append(config.target)

    continuous_scalers = {}
    for var in continuous_vars:
        if var in train_data.columns:
            scaler = TimeSeriesScaler(method='standard')
            scaler.fit(train_data[var].values.reshape(-1, 1))
            continuous_scalers[var] = scaler

            # Transform all datasets
            train_data[var] = scaler.transform(
                train_data[var].values.reshape(-1, 1)
            ).flatten()
            val_data[var] = scaler.transform(
                val_data[var].values.reshape(-1, 1)
            ).flatten()
            if test_data is not None:
                test_data[var] = scaler.transform(
                    test_data[var].values.reshape(-1, 1)
                ).flatten()

    # Create datasets
    train_dataset = TimeSeriesDataset(
        data=train_data,
        config=config,
        continuous_scalers=continuous_scalers,
        mode='train',
    )

    val_dataset = TimeSeriesDataset(
        data=val_data,
        config=config,
        continuous_scalers=continuous_scalers,
        mode='val',
    )

    test_dataset = None
    if test_data is not None:
        test_dataset = TimeSeriesDataset(
            data=test_data,
            config=config,
            continuous_scalers=continuous_scalers,
            mode='test',
        )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
        )

    return train_loader, val_loader, test_loader


class GroupedTimeSeriesDataset(TimeSeriesDataset):
    """
    Extended TimeSeriesDataset that handles grouped time series.

    Useful for datasets with multiple entities (e.g., multiple stores)
    where each group has its own time series.

    Args:
        data: DataFrame with time series data
        config: TFTConfig with model configuration
        group_col: Column name for group identifiers
        static_scalers: Optional dict of scalers for static features
        continuous_scalers: Optional dict of scalers for continuous features
        mode: 'train', 'val', or 'test'
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config,
        group_col: str,
        static_scalers: Optional[Dict] = None,
        continuous_scalers: Optional[Dict] = None,
        mode: str = 'train',
    ):
        self.group_col = group_col
        super().__init__(
            data=data,
            config=config,
            static_scalers=static_scalers,
            continuous_scalers=continuous_scalers,
            mode=mode,
        )

    def _create_valid_indices(self) -> List[Tuple[str, int]]:
        """
        Create valid indices for grouped time series.

        Returns list of (group_id, start_idx) tuples.
        """
        valid_indices = []

        for group_id, group_data in self.data.groupby(self.group_col):
            n = len(group_data)
            group_start_idx = group_data.index[0]

            for i in range(n - self.total_length + 1):
                valid_indices.append((group_id, group_start_idx + i))

        return valid_indices

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single windowed sample from grouped data."""
        group_id, start_idx = self.valid_indices[idx]

        # Get group data
        group_data = self.data[self.data[self.group_col] == group_id]

        # Find local index within group
        local_start = list(group_data.index).index(start_idx)
        local_end = local_start + self.total_length

        # Get window
        window_data = group_data.iloc[local_start:local_end]

        # Use parent class logic for creating batch
        # Temporarily override self.data with window context
        original_data = self.data
        self.data = group_data
        original_indices = self.valid_indices
        self.valid_indices = [local_start]

        try:
            batch = super().__getitem__(0)
        finally:
            self.data = original_data
            self.valid_indices = original_indices

        return batch
