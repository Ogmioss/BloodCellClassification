#!/usr/bin/env python3
"""Script de diagnostic pour vérifier le device PyTorch"""

import torch
from pathlib import Path
import sys
import os

# Disable oneDNN to fix "could not create a primitive" error
torch.backends.mkldnn.enabled = False

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.models.model_factory import ModelFactory

print("=" * 60)
print("DIAGNOSTIC PYTORCH DEVICE")
print("=" * 60)

print(f"\nMPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

print(f"\nModelFactory.get_device(): {ModelFactory.get_device()}")

# Test création d'un tensor
device = ModelFactory.get_device()
print(f"\nCréation d'un tensor sur device '{device}':")
test_tensor = torch.randn(1, 3, 224, 224).to(device)
print(f"Tensor device: {test_tensor.device}")

# Test création d'un modèle simple
print(f"\nCréation d'un modèle simple:")
model = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3).to(device)
print(f"Model device (premier paramètre): {next(model.parameters()).device}")

# Test forward pass
print(f"\nTest forward pass:")
try:
    output = model(test_tensor)
    print(f"✅ Forward pass réussi!")
    print(f"Output device: {output.device}")
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Forward pass échoué: {e}")

print("\n" + "=" * 60)
