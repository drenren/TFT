"""
Training loop for Temporal Fusion Transformer.

This module provides the main Trainer class that handles the complete
training workflow including optimization, validation, and callbacks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any
from tqdm import tqdm

from .losses import QuantileLoss
from .metrics import MetricsCalculator
from .callbacks import CallbackList, Callback


class TFTTrainer:
    """
    Trainer class for Temporal Fusion Transformer.

    Handles the training loop, validation, optimization, and callbacks.

    Args:
        model: TFT model instance
        config: TFTConfig object
        optimizer: Optional optimizer (if None, creates Adam)
        loss_fn: Optional loss function (if None, uses QuantileLoss)
        device: Device for training
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config

        # Setup device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Setup loss function
        if loss_fn is None:
            self.loss_fn = QuantileLoss(
                quantiles=config.quantiles,
                reduction='mean',
            )
        else:
            self.loss_fn = loss_fn
        self.loss_fn.to(self.device)

        # Setup metrics calculator
        self.metrics_calculator = MetricsCalculator(
            quantiles=config.quantiles,
            device=self.device,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (if None, uses config.max_epochs)
            callbacks: List of callback instances
        """
        if epochs is None:
            epochs = self.config.max_epochs

        # Setup callbacks
        if callbacks is None:
            callbacks = []
        callback_list = CallbackList(callbacks)

        # Start training
        callback_list.on_train_start(model=self.model)

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Epoch start
            callback_list.on_epoch_start(epoch, model=self.model)

            # Training
            train_metrics = self.train_epoch(train_loader, callback_list)

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = {}

            # Combine metrics
            metrics = {
                'train_loss': train_metrics['loss'],
                **{f'train_{k}': v for k, v in train_metrics.items() if k != 'loss'},
            }
            if val_metrics:
                metrics['val_loss'] = val_metrics['loss']
                metrics.update({f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'})

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])

            # Epoch end
            callback_list.on_epoch_end(
                epoch,
                metrics,
                model=self.model,
                optimizer=self.optimizer,
            )

            # Check early stopping
            if callback_list.should_stop_training():
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # End training
        callback_list.on_train_end(model=self.model)

    def train_epoch(
        self,
        train_loader: DataLoader,
        callback_list: CallbackList,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            callback_list: Callbacks

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        self.metrics_calculator.reset()

        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')

        for batch_idx, batch in enumerate(pbar):
            callback_list.on_batch_start(batch_idx)

            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)

            # Compute loss
            predictions = outputs['predictions']
            targets = batch['target']
            loss = self.loss_fn(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val,
                )

            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update metrics calculator
            self.metrics_calculator.update(predictions, targets)

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            callback_list.on_batch_end(batch_idx)

        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        metrics = self.metrics_calculator.compute_and_reset()
        metrics['loss'] = avg_loss

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0

        self.metrics_calculator.reset()

        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(batch)

            # Compute loss
            predictions = outputs['predictions']
            targets = batch['target']
            loss = self.loss_fn(predictions, targets)

            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1

            # Update metrics calculator
            self.metrics_calculator.update(predictions, targets)

        # Compute metrics
        avg_loss = epoch_loss / num_batches
        metrics = self.metrics_calculator.compute_and_reset()
        metrics['loss'] = avg_loss

        return metrics

    @torch.no_grad()
    def predict(
        self,
        data_loader: DataLoader,
        return_attention: bool = False,
        return_variable_selection: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions on a dataset.

        Args:
            data_loader: Data loader
            return_attention: Whether to return attention weights
            return_variable_selection: Whether to return variable selection weights

        Returns:
            Dictionary containing predictions and optionally interpretability info
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_attention = []
        all_encoder_vsn = []
        all_decoder_vsn = []

        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(batch)

            # Collect outputs
            all_predictions.append(outputs['predictions'].cpu())
            all_targets.append(batch['target'].cpu())

            if return_attention and 'attention_weights' in outputs:
                all_attention.append(outputs['attention_weights'].cpu())

            if return_variable_selection:
                if 'encoder_variable_selection' in outputs:
                    all_encoder_vsn.append(outputs['encoder_variable_selection'].cpu())
                if 'decoder_variable_selection' in outputs:
                    all_decoder_vsn.append(outputs['decoder_variable_selection'].cpu())

        # Concatenate
        result = {
            'predictions': torch.cat(all_predictions, dim=0),
            'targets': torch.cat(all_targets, dim=0),
        }

        if return_attention and all_attention:
            result['attention_weights'] = torch.cat(all_attention, dim=0)

        if return_variable_selection:
            if all_encoder_vsn:
                result['encoder_variable_selection'] = torch.cat(all_encoder_vsn, dim=0)
            if all_decoder_vsn:
                result['decoder_variable_selection'] = torch.cat(all_decoder_vsn, dim=0)

        return result

    def save_checkpoint(self, filepath: str, **extra_info):
        """
        Save a training checkpoint.

        Args:
            filepath: Path to save checkpoint
            extra_info: Additional info to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            **extra_info,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """
        Load a training checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
