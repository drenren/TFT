"""
Device detection and management utilities for TFT.

This module provides utilities for detecting and selecting the best
available device (CUDA GPU, Apple MPS, or CPU).
"""

import torch


def get_device(preferred_device: str = "auto") -> torch.device:
    """
    Get the best available device for PyTorch operations.

    Checks in order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon GPUs on macOS)
    3. CPU (fallback)

    Args:
        preferred_device: One of "auto", "cuda", "mps", "cpu"
                         If "auto", selects best available device

    Returns:
        torch.device object for the selected device

    Examples:
        >>> device = get_device()  # Automatically selects best device
        >>> device = get_device("cuda")  # Prefer CUDA if available
        >>> device = get_device("cpu")  # Force CPU usage
    """
    if preferred_device == "auto":
        # Check CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA GPU: {device_name}")
            return device

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS (Metal Performance Shaders)")
            return device

        # Fallback to CPU
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")
        return device

    elif preferred_device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA GPU: {device_name}")
            return device
        else:
            print("Warning: CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")

    elif preferred_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS (Metal Performance Shaders)")
            return device
        else:
            print("Warning: MPS requested but not available, falling back to CPU")
            return torch.device("cpu")

    elif preferred_device == "cpu":
        return torch.device("cpu")

    else:
        raise ValueError(
            f"Invalid device: {preferred_device}. "
            "Must be one of: 'auto', 'cuda', 'mps', 'cpu'"
        )


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        Dictionary with device availability and details
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu_available": True,
    }

    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda

    if info["mps_available"]:
        info["mps_built"] = torch.backends.mps.is_built()

    info["pytorch_version"] = torch.__version__

    return info


def print_device_info():
    """Print detailed information about available devices."""
    info = get_device_info()

    print("=" * 60)
    print("PyTorch Device Information")
    print("=" * 60)
    print(f"PyTorch version: {info['pytorch_version']}")
    print()

    print("Available Devices:")
    print(f"  CUDA (NVIDIA GPU): {'✓ Available' if info['cuda_available'] else '✗ Not available'}")
    if info["cuda_available"]:
        print(f"    - Device count: {info['cuda_device_count']}")
        print(f"    - Device name: {info['cuda_device_name']}")
        print(f"    - CUDA version: {info['cuda_version']}")

    print(f"  MPS (Apple Silicon): {'✓ Available' if info['mps_available'] else '✗ Not available'}")
    if info["mps_available"]:
        print(f"    - Built: {info['mps_built']}")

    print(f"  CPU: ✓ Available")
    print("=" * 60)


def move_to_device(obj, device: torch.device):
    """
    Move tensors or models to the specified device.

    Handles both individual tensors and dictionaries of tensors.

    Args:
        obj: Tensor, model, or dict of tensors to move
        device: Target device

    Returns:
        Object moved to the specified device
    """
    if isinstance(obj, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [v.to(device) if isinstance(v, torch.Tensor) else v
                for v in obj]
    elif hasattr(obj, 'to'):
        return obj.to(device)
    else:
        return obj
