"""
Data preprocessing utilities for time series.

This module provides functions for preparing time series data for TFT training,
including time feature extraction and train/validation/test splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from datetime import datetime


def extract_time_features(
    timestamps: pd.Series,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract time-based features from timestamps.

    Args:
        timestamps: Pandas Series of timestamps
        features: List of features to extract. If None, extracts all.
                 Options: 'hour', 'day_of_week', 'day_of_month', 'day_of_year',
                         'week_of_year', 'month', 'quarter', 'year',
                         'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                         'week_sin', 'week_cos', 'month_sin', 'month_cos'

    Returns:
        DataFrame with extracted time features
    """
    timestamps = pd.to_datetime(timestamps)

    if features is None:
        features = [
            'hour', 'day_of_week', 'day_of_month', 'month',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        ]

    time_features = {}

    # Basic features
    if 'hour' in features:
        time_features['hour'] = timestamps.dt.hour
    if 'day_of_week' in features:
        time_features['day_of_week'] = timestamps.dt.dayofweek
    if 'day_of_month' in features:
        time_features['day_of_month'] = timestamps.dt.day
    if 'day_of_year' in features:
        time_features['day_of_year'] = timestamps.dt.dayofyear
    if 'week_of_year' in features:
        time_features['week_of_year'] = timestamps.dt.isocalendar().week
    if 'month' in features:
        time_features['month'] = timestamps.dt.month
    if 'quarter' in features:
        time_features['quarter'] = timestamps.dt.quarter
    if 'year' in features:
        time_features['year'] = timestamps.dt.year

    # Cyclical features (encoded as sin/cos for periodicity)
    if 'hour_sin' in features:
        time_features['hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
    if 'hour_cos' in features:
        time_features['hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24)

    if 'day_sin' in features:
        time_features['day_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
    if 'day_cos' in features:
        time_features['day_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)

    if 'week_sin' in features:
        week = timestamps.dt.isocalendar().week.astype(float)
        time_features['week_sin'] = np.sin(2 * np.pi * week / 52)
    if 'week_cos' in features:
        week = timestamps.dt.isocalendar().week.astype(float)
        time_features['week_cos'] = np.cos(2 * np.pi * week / 52)

    if 'month_sin' in features:
        time_features['month_sin'] = np.sin(2 * np.pi * timestamps.dt.month / 12)
    if 'month_cos' in features:
        time_features['month_cos'] = np.cos(2 * np.pi * timestamps.dt.month / 12)

    return pd.DataFrame(time_features, index=timestamps.index)


def create_time_series_splits(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    time_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets.

    Splits are made chronologically (no shuffling to preserve temporal order).

    Args:
        data: DataFrame with time series data
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        time_col: Name of time column to sort by (if not already sorted)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Sort by time if specified
    if time_col is not None:
        data = data.sort_values(time_col)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]

    return train_df, val_df, test_df


def create_group_time_series_splits(
    data: pd.DataFrame,
    group_col: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    time_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data by groups (e.g., different stores).

    Each group is split independently to maintain temporal order within groups.

    Args:
        data: DataFrame with time series data
        group_col: Column name for group identifiers
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        time_col: Name of time column to sort by

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for group, group_data in data.groupby(group_col):
        train_df, val_df, test_df = create_time_series_splits(
            group_data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            time_col=time_col,
        )
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)

    train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
    val_df = pd.concat(val_dfs, axis=0, ignore_index=True)
    test_df = pd.concat(test_dfs, axis=0, ignore_index=True)

    return train_df, val_df, test_df


def normalize_columns(
    data: pd.DataFrame,
    columns: List[str],
    method: str = 'standard',
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Normalize specified columns in a DataFrame.

    Args:
        data: Input DataFrame
        columns: List of columns to normalize
        method: Normalization method ('standard' or 'minmax')

    Returns:
        Tuple of (normalized_df, normalization_params)
    """
    data = data.copy()
    params = {}

    for col in columns:
        if method == 'standard':
            mean = data[col].mean()
            std = data[col].std()
            std = std if std > 0 else 1.0
            data[col] = (data[col] - mean) / std
            params[col] = (mean, std)

        elif method == 'minmax':
            min_val = data[col].min()
            max_val = data[col].max()
            range_val = max_val - min_val
            range_val = range_val if range_val > 0 else 1.0
            data[col] = (data[col] - min_val) / range_val
            params[col] = (min_val, max_val)

    return data, params


def fill_missing_values(
    data: pd.DataFrame,
    method: str = 'ffill',
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fill missing values in time series data.

    Args:
        data: Input DataFrame
        method: Method for filling ('ffill', 'bfill', 'interpolate', 'zero', 'mean')
        columns: Columns to fill (if None, fills all columns)

    Returns:
        DataFrame with filled values
    """
    data = data.copy()

    if columns is None:
        columns = data.columns.tolist()

    for col in columns:
        if method == 'ffill':
            data[col] = data[col].fillna(method='ffill')
            data[col] = data[col].fillna(method='bfill')  # Handle leading NaNs
        elif method == 'bfill':
            data[col] = data[col].fillna(method='bfill')
            data[col] = data[col].fillna(method='ffill')  # Handle trailing NaNs
        elif method == 'interpolate':
            data[col] = data[col].interpolate(method='linear')
            data[col] = data[col].fillna(method='bfill')
            data[col] = data[col].fillna(method='ffill')
        elif method == 'zero':
            data[col] = data[col].fillna(0)
        elif method == 'mean':
            data[col] = data[col].fillna(data[col].mean())

    return data


def create_lag_features(
    data: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create lagged features for time series.

    Args:
        data: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag values (e.g., [1, 2, 7, 14])
        group_col: Optional group column for grouped lagging

    Returns:
        DataFrame with lag features added
    """
    data = data.copy()

    for col in columns:
        for lag in lags:
            lag_col_name = f"{col}_lag_{lag}"

            if group_col is not None:
                data[lag_col_name] = data.groupby(group_col)[col].shift(lag)
            else:
                data[lag_col_name] = data[col].shift(lag)

    return data


def create_rolling_features(
    data: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    functions: List[str] = ['mean', 'std'],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create rolling window features for time series.

    Args:
        data: Input DataFrame
        columns: Columns to create rolling features for
        windows: List of window sizes (e.g., [7, 14, 30])
        functions: List of aggregation functions ('mean', 'std', 'min', 'max', 'sum')
        group_col: Optional group column for grouped rolling

    Returns:
        DataFrame with rolling features added
    """
    data = data.copy()

    for col in columns:
        for window in windows:
            for func in functions:
                feature_name = f"{col}_rolling_{window}_{func}"

                if group_col is not None:
                    grouped = data.groupby(group_col)[col].rolling(
                        window=window, min_periods=1
                    )
                    if func == 'mean':
                        data[feature_name] = grouped.mean().reset_index(level=0, drop=True)
                    elif func == 'std':
                        data[feature_name] = grouped.std().reset_index(level=0, drop=True)
                    elif func == 'min':
                        data[feature_name] = grouped.min().reset_index(level=0, drop=True)
                    elif func == 'max':
                        data[feature_name] = grouped.max().reset_index(level=0, drop=True)
                    elif func == 'sum':
                        data[feature_name] = grouped.sum().reset_index(level=0, drop=True)
                else:
                    rolling = data[col].rolling(window=window, min_periods=1)
                    if func == 'mean':
                        data[feature_name] = rolling.mean()
                    elif func == 'std':
                        data[feature_name] = rolling.std()
                    elif func == 'min':
                        data[feature_name] = rolling.min()
                    elif func == 'max':
                        data[feature_name] = rolling.max()
                    elif func == 'sum':
                        data[feature_name] = rolling.sum()

    return data
