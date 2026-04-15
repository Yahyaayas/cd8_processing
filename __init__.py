"""
CD8 Tile Processing Pipeline

Generates paired H&E/IHC tiles with CD8 masks from WSI data.
"""

from . import config
from . import utils
from . import registration
from . import classpose_wrapper
from . import tile_generator

__version__ = "1.0.0"
