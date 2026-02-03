"""Compatibility shim so scripts can import `from models import get_backbone`.

This file loads the real backbone implementation from
`contrastive-fruits/ResNet.py` and re-exports `get_backbone`.

It avoids modifying many local imports that expect a top-level
`models` module in older scripts.
"""
import importlib.util
import os

_here = os.path.dirname(__file__)
_resnet_path = os.path.join(_here, 'contrastive-fruits', 'ResNet.py')
if os.path.exists(_resnet_path):
    spec = importlib.util.spec_from_file_location('cf_resnet', _resnet_path)
    _cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_cf)
    # re-export get_backbone if present
    if hasattr(_cf, 'get_backbone'):
        get_backbone = getattr(_cf, 'get_backbone')
    else:
        raise ImportError(f'get_backbone not found in {_resnet_path}')
else:
    raise ImportError(f'ResNet implementation not found at {_resnet_path}')
