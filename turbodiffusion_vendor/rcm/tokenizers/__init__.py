"""
Vendored TurboDiffusion code for ComfyUI TurboDiffusion custom node.

This package contains essential code from TurboDiffusion (https://github.com/thu-ml/TurboDiffusion)
to eliminate external dependencies and simplify installation.

License: Apache 2.0 (same as TurboDiffusion)
"""

import sys
import os

# Add this directory to sys.path so that turbodiffusion modules can import each other
# This allows the vendored code to use absolute imports like "from rcm.networks import..."
_vendor_dir = os.path.dirname(os.path.abspath(__file__))
if _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

__version__ = "vendored-from-turbodiffusion"
