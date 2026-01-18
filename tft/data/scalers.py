"""
Scalers for time series data normalization.

This module provides scalers specifically designed for time series forecasting,
ensuring proper train/test splitting and handling of temporal data.
"""

import numpy as np
import torch
from typing import Optional, Union, List
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


class TimeSeriesScaler:
    """
    Scaler for time series data with per-feature normalization.

    This scaler fits on training data only and transforms both
    training and validation/test data using the same statistics.

    Args:
        method: Normalization method ('standard', 'minmax', or 'robust')
    """

    def __init__(self, method: str = 'standard'):
        self.method = method
        self.is_fitted = False

        if method == 'standard':
            self.mean_ = None
            self.std_ = None
        elif method == 'minmax':
            self.min_ = None
            self.max_ = None
        elif method == 'robust':
            self.median_ = None
            self.iqr_ = None
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'TimeSeriesScaler':
        """
        Fit the scaler on training data.

        Args:
            data: Training data of shape (num_samples, num_features)
                  or (num_samples, time_steps, num_features)

        Returns:
            self
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # Reshape if 3D (flatten time dimension)
        if data.ndim == 3:
            original_shape = data.shape
            data = data.reshape(-1, data.shape[-1])

        if self.method == 'standard':
            self.mean_ = np.mean(data, axis=0, keepdims=True)
            self.std_ = np.std(data, axis=0, keepdims=True)
            # Avoid division by zero
            self.std_[self.std_ == 0] = 1.0

        elif self.method == 'minmax':
            self.min_ = np.min(data, axis=0, keepdims=True)
            self.max_ = np.max(data, axis=0, keepdims=True)
            # Avoid division by zero
            range_ = self.max_ - self.min_
            range_[range_ == 0] = 1.0
            self.max_ = self.min_ + range_

        elif self.method == 'robust':
            self.median_ = np.median(data, axis=0, keepdims=True)
            q75, q25 = np.percentile(data, [75, 25], axis=0, keepdims=True)
            self.iqr_ = q75 - q25
            # Avoid division by zero
            self.iqr_[self.iqr_ == 0] = 1.0

        self.is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data using fitted statistics.

        Args:
            data: Data to transform

        Returns:
            Transformed data with same type and shape as input
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            dtype = data.dtype
            data = data.cpu().numpy()

        if self.method == 'standard':
            transformed = (data - self.mean_) / self.std_

        elif self.method == 'minmax':
            transformed = (data - self.min_) / (self.max_ - self.min_)

        elif self.method == 'robust':
            transformed = (data - self.median_) / self.iqr_

        if is_torch:
            transformed = torch.tensor(transformed, dtype=dtype, device=device)

        return transformed

    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverse transform normalized data back to original scale.

        Args:
            data: Normalized data to inverse transform

        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            dtype = data.dtype
            data = data.cpu().numpy()

        if self.method == 'standard':
            original = data * self.std_ + self.mean_

        elif self.method == 'minmax':
            original = data * (self.max_ - self.min_) + self.min_

        elif self.method == 'robust':
            original = data * self.iqr_ + self.median_

        if is_torch:
            original = torch.tensor(original, dtype=dtype, device=device)

        return original

    def fit_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Fit the scaler and transform the data in one step.

        Args:
            data: Data to fit and transform

        Returns:
            Transformed data
        """
        self.fit(data)
        return self.transform(data)


class GroupScaler:
    """
    Scaler that maintains separate statistics for different groups.

    Useful when you have multiple time series (e.g., different stores)
    and want to normalize each group independently.

    Args:
        method: Normalization method ('standard', 'minmax', or 'robust')
    """

    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scalers = {}
        self.is_fitted = False

    def fit(
        self,
        data: Union[np.ndarray, torch.Tensor],
        groups: Union[np.ndarray, List],
    ) -> 'GroupScaler':
        """
        Fit separate scalers for each group.

        Args:
            data: Training data
            groups: Group identifiers for each sample

        Returns:
            self
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        if isinstance(groups, torch.Tensor):
            groups = groups.numpy()

        groups = np.array(groups)
        unique_groups = np.unique(groups)

        for group in unique_groups:
            mask = groups == group
            group_data = data[mask]

            scaler = TimeSeriesScaler(method=self.method)
            scaler.fit(group_data)
            self.scalers[group] = scaler

        self.is_fitted = True
        return self

    def transform(
        self,
        data: Union[np.ndarray, torch.Tensor],
        groups: Union[np.ndarray, List],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data using group-specific scalers.

        Args:
            data: Data to transform
            groups: Group identifiers

        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            dtype = data.dtype
            data = data.cpu().numpy()

        if isinstance(groups, torch.Tensor):
            groups = groups.cpu().numpy()

        groups = np.array(groups)
        transformed = np.zeros_like(data)

        for group, scaler in self.scalers.items():
            mask = groups == group
            if mask.any():
                transformed[mask] = scaler.transform(data[mask])

        if is_torch:
            transformed = torch.tensor(transformed, dtype=dtype, device=device)

        return transformed

    def inverse_transform(
        self,
        data: Union[np.ndarray, torch.Tensor],
        groups: Union[np.ndarray, List],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverse transform data using group-specific scalers.

        Args:
            data: Normalized data
            groups: Group identifiers

        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            dtype = data.dtype
            data = data.cpu().numpy()

        if isinstance(groups, torch.Tensor):
            groups = groups.cpu().numpy()

        groups = np.array(groups)
        original = np.zeros_like(data)

        for group, scaler in self.scalers.items():
            mask = groups == group
            if mask.any():
                original[mask] = scaler.inverse_transform(data[mask])

        if is_torch:
            original = torch.tensor(original, dtype=dtype, device=device)

        return original
