"""
Test script to verify device detection on different systems.

This script tests that TFT properly detects and uses:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)
"""

import sys
sys.path.append('..')

import torch
from tft.utils import get_device, print_device_info, get_device_info

def main():
    print("=" * 80)
    print("TFT Device Detection Test")
    print("=" * 80)
    print()

    # Print device information
    print_device_info()
    print()

    # Test automatic device selection
    print("Testing automatic device selection...")
    device = get_device("auto")
    print(f"Selected device: {device}")
    print()

    # Get device info as dict
    info = get_device_info()
    print("Device Info Dictionary:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Test tensor operations on selected device
    print("Testing tensor operations on selected device...")
    try:
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.matmul(x, y)
        print(f"✓ Matrix multiplication successful on {device}")
        print(f"  Result shape: {z.shape}")
        print(f"  Result device: {z.device}")
    except Exception as e:
        print(f"✗ Error during tensor operations: {e}")
    print()

    # Test each device type if requested
    print("Testing specific device requests:")
    for dev_type in ["auto", "cuda", "mps", "cpu"]:
        try:
            dev = get_device(dev_type)
            print(f"  {dev_type:6s} -> {dev}")
        except Exception as e:
            print(f"  {dev_type:6s} -> Error: {e}")
    print()

    # Recommendations
    print("=" * 80)
    print("Recommendations:")
    print("=" * 80)
    if info["cuda_available"]:
        print("✓ NVIDIA GPU detected - Use 'cuda' or 'auto' for best performance")
    elif info["mps_available"]:
        print("✓ Apple Silicon detected - Use 'mps' or 'auto' for GPU acceleration")
    else:
        print("! No GPU detected - Using CPU (training will be slower)")
    print()
    print("For TFT training, the device is automatically selected when you")
    print("create a TFTTrainer or use get_device('auto').")
    print("=" * 80)


if __name__ == '__main__':
    main()
