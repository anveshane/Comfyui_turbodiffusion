"""
Vendored TurboDiffusion code for ComfyUI.

DO NOT perform strict path checks here.
ComfyUI already controls sys.path.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure vendor root is importable
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

__version__ = "vendored-turbodiffusion"
