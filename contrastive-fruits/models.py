"""Compatibility shim so scripts can import `from models import get_backbone`.

This file loads the real backbone implementation from
`contrastive-fruits/ResNet.py` and re-exports `get_backbone`.

It avoids modifying many local imports that expect a top-level
`models` module in older scripts.
"""
import importlib.util
import os

_here = os.path.dirname(__file__)
# Search common locations for the ResNet implementation to support different repo layouts
_candidates = [
    os.path.join(_here, 'ResNet.py'),
    os.path.join(_here, 'contrastive-fruits', 'ResNet.py')
]
_resnet_path = None
for p in _candidates:
    if os.path.exists(p):
        _resnet_path = p
        break
if _resnet_path is None:
    raise ImportError(f'ResNet implementation not found; tried: {", ".join(_candidates)}')

spec = importlib.util.spec_from_file_location('cf_resnet', _resnet_path)
_cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_cf)
# re-export get_backbone if present
if hasattr(_cf, 'get_backbone'):
    get_backbone = getattr(_cf, 'get_backbone')
else:
    raise ImportError(f'get_backbone not found in {_resnet_path}')
