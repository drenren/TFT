"""
Training callbacks for Temporal Fusion Transformer.

This module provides callbacks for early stopping, model checkpointing,
and learning rate scheduling during training.
"""

import torch
import os
from typing import Optional, Dict, Any
from pathlib import Path


class Callback:
    """Base callback class."""

    def on_epoch_start(self, epoch: int, **kwargs):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_train_start(self, **kwargs):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, **kwargs):
        """Called at the end of training."""
        pass

    def on_batch_start(self, batch_idx: int, **kwargs):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch_idx: int, **kwargs):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.

    Stops training when a monitored metric has stopped improving.

    Args:
        monitor: Metric to monitor (e.g., 'val_loss')
        patience: Number of epochs with no improvement to wait
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' or 'max' - whether lower or higher is better
        verbose: Whether to print messages
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.should_stop = False

        if mode == 'min':
            self.monitor_op = lambda x, y: x < y - min_delta
            self.best_score = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda x, y: x > y + min_delta
            self.best_score = float('-inf')
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """Check if training should stop."""
        if self.monitor not in metrics:
            if self.verbose:
                print(f"Warning: Metric '{self.monitor}' not found in metrics")
            return

        current_score = metrics[self.monitor]

        if self.monitor_op(current_score, self.best_score):
            # Improvement
            self.best_score = current_score
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Metric improved to {current_score:.6f}")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training after {epoch + 1} epochs")


class ModelCheckpoint(Callback):
    """
    Model checkpoint callback.

    Saves model checkpoints during training.

    Args:
        filepath: Path to save checkpoints (can include formatting like 'model_{epoch}.pth')
        monitor: Metric to monitor for best model
        save_best_only: Whether to only save the best model
        mode: 'min' or 'max'
        verbose: Whether to print messages
    """

    def __init__(
        self,
        filepath: str = 'checkpoint.pth',
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min',
        verbose: bool = True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose

        if mode == 'min':
            self.monitor_op = lambda x, y: x < y
            self.best_score = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda x, y: x > y
            self.best_score = float('-inf')
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ):
        """Save checkpoint if conditions are met."""
        if model is None:
            return

        current_score = metrics.get(self.monitor, None)

        if current_score is None:
            if self.verbose:
                print(f"Warning: Metric '{self.monitor}' not found, saving anyway")
            should_save = not self.save_best_only
        else:
            should_save = self.monitor_op(current_score, self.best_score) or not self.save_best_only
            if self.monitor_op(current_score, self.best_score):
                self.best_score = current_score

        if should_save:
            # Format filepath with epoch number
            filepath = self.filepath.format(epoch=epoch, **metrics)

            # Create checkpoint dictionary
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
            }

            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            # Save
            torch.save(checkpoint, filepath)

            if self.verbose:
                print(f"ModelCheckpoint: Saved checkpoint to {filepath}")


class LearningRateScheduler(Callback):
    """
    Learning rate scheduler callback.

    Adjusts learning rate based on training progress.

    Args:
        scheduler: PyTorch learning rate scheduler
        monitor: Metric to monitor (for ReduceLROnPlateau)
        verbose: Whether to print messages
    """

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: Optional[str] = None,
        verbose: bool = True,
    ):
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """Update learning rate."""
        # For ReduceLROnPlateau, we need to pass the metric value
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self.monitor is None:
                raise ValueError("monitor must be specified for ReduceLROnPlateau")

            if self.monitor in metrics:
                self.scheduler.step(metrics[self.monitor])
            else:
                print(f"Warning: Metric '{self.monitor}' not found for LR scheduler")
        else:
            # For other schedulers, just step
            self.scheduler.step()

        if self.verbose:
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6e}")


class MetricsLogger(Callback):
    """
    Callback for logging metrics to console and/or file.

    Args:
        log_file: Optional file path to log metrics
        verbose: Whether to print to console
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        verbose: bool = True,
    ):
        self.log_file = log_file
        self.verbose = verbose

        if log_file is not None:
            # Create log file and write header
            with open(log_file, 'w') as f:
                f.write("epoch,")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """Log metrics."""
        if self.verbose:
            metrics_str = ', '.join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            print(f"Epoch {epoch + 1} - {metrics_str}")

        if self.log_file is not None:
            # Append to log file
            with open(self.log_file, 'a') as f:
                # Write metrics values
                values = [str(epoch + 1)] + [f"{v:.6f}" for v in metrics.values()]
                f.write(','.join(values) + '\n')


class TensorBoardLogger(Callback):
    """
    Callback for logging metrics to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs
    """

    def __init__(self, log_dir: str = 'runs'):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """Log metrics to TensorBoard."""
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, epoch)

    def on_train_end(self, **kwargs):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class CallbackList:
    """
    Container for managing multiple callbacks.

    Args:
        callbacks: List of callback instances
    """

    def __init__(self, callbacks: list):
        self.callbacks = callbacks

    def on_epoch_start(self, epoch: int, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, **kwargs)

    def on_train_start(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_start(**kwargs)

    def on_train_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_batch_start(self, batch_idx: int, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_start(batch_idx, **kwargs)

    def on_batch_end(self, batch_idx: int, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, **kwargs)

    def should_stop_training(self) -> bool:
        """Check if any callback requests to stop training."""
        return any(
            getattr(callback, 'should_stop', False)
            for callback in self.callbacks
        )
