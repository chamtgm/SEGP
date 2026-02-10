#!/usr/bin/env python3
"""Convert directory-based checkpoint to .pt file"""
import torch
import os
import pickle
from pathlib import Path
from typing import Any

# Custom unpickler for PyTorch
class TorchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch._utils':
            return super().find_class('torch', name)
        return super().find_class(module, name)

# Source directory and output file
CKPT_DIR = Path(__file__).parent / 'ckpt_epoch_1000'
OUTPUT_PT = Path(__file__).parent / 'ckpt_epoch_1000.pt'

print(f"Converting checkpoint from {CKPT_DIR}")

try:
    # List directory contents
    print("\nDirectory contents:")
    for item in sorted(CKPT_DIR.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024*1024)
            print(f"  {item.name} ({size_mb:.1f} MB)")
        else:
            print(f"  {item.name}/ (directory)")
    
    # Try loading with BytesIO and custom handling
    print("\nAttempting to load checkpoint...")
    pkl_file = CKPT_DIR / 'data.pkl'
    
    # Read and deserialize with persistent_load handling
    def persistent_load(pid):
        print(f"  persistent_id: {pid}")
        # Return a placeholder - this is for storage references
        return f"<persistent:{pid}>"
    
    with open(pkl_file, 'rb') as f:
        ckpt = torch.load(f, map_location='cpu', weights_only=False, pickle_load_args={'persistent_load': persistent_load})
    
    print(f"✓ Loaded checkpoint type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"  Keys: {list(ckpt.keys())}")
    
    # Save as .pt file
    print(f"\nSaving to {OUTPUT_PT}...")
    torch.save(ckpt, str(OUTPUT_PT))
    print(f"✓ Successfully converted! File size: {OUTPUT_PT.stat().st_size / (1024*1024):.1f} MB")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
