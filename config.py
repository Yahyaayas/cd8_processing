"""
Configuration for CD8 Tile Processing Pipeline
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Workspace root
WORKSPACE = Path(r"/home/n207/project-Yahya/image-processing")

# Input paths
DATA_ROOT = Path(r"/home/n207/project-Yahya/data")
RAW_DATA_DIR = Path(r"/home/n207/project-Yahya/data/WSI")
IHC_LABEL_DIR = DATA_ROOT / "ihc_stain_label"

# Output paths
PROCESSED_DIR = WORKSPACE / "processed_test"
HE_TILES_DIR = PROCESSED_DIR / "he_tiles"
IHC_TILES_DIR = PROCESSED_DIR / "ihc_tiles"
MASK_TILES_DIR = PROCESSED_DIR / "mask_tiles"

# Classpose setup
CLASSPOSE_WORKDIR = Path.home() / "classpose_workspace"
CLASSPOSE_REPO = CLASSPOSE_WORKDIR / "classpose"
CLASSPOSE_MODELS = CLASSPOSE_WORKDIR / "models"

# ============================================================================
# REGISTRATION PARAMETERS
# ============================================================================

REG_LEVEL = 2              # Pyramid level for global registration
SIFT_NFEATURES = 5000      # Number of SIFT features
LOWE_RATIO = 0.75          # Lowe's ratio for feature matching
RANSAC_THRESHOLD = 3.0     # RANSAC reprojection threshold
RANSAC_CONFIDENCE = 0.9995 # RANSAC confidence
RANSAC_MAX_ITERS = 5000    # RANSAC max iterations

# ============================================================================
# IOU MATCHING
# ============================================================================

IOU_THRESHOLD = 0.5        # IoU threshold for CD8+ classification

# ============================================================================
# TILE PARAMETERS
# ============================================================================

TILE_SIZE = 1024           # Tile size (pixels)
OVERLAP = 0                # Tile overlap (for grid-based, not used for CD8+ centered)
MAX_TILE = 8000            # Max tile generated

# ============================================================================
# TILE FILTERING
# ============================================================================

BACKGROUND_THRESHOLD = 0.8  # Max white-pixel ratio
BLACK_THRESHOLD = 0.8       # Max black-pixel ratio
GREY_THRESHOLD = 17         # Min std-dev (texture)
RGB_MEAN_MIN = 100          # Min mean RGB intensity

# ============================================================================
# CLASSPOSE PARAMETERS
# ============================================================================

CLASSPOSE_MODEL = "puma"           # Model config
CLASSPOSE_DEVICE = "cuda:0"        # Device
CLASSPOSE_BATCH_SIZE = 8           # Batch size
CLASSPOSE_BF16 = True              # Use BF16 (Ampere+ GPU)
CLASSPOSE_TILE_SIZE = 1024         # Tile size for Classpose
CLASSPOSE_OVERLAP = 64             # Overlap for Classpose

# ============================================================================
# CLASSPOSE LYMPHOCYTE FILTER
# ============================================================================

LYMPHOCYTE_CLASS = "Lymphocyte"    # Class name used for CD8+ IoU matching

# ============================================================================
# PUMA CLASS MAPPING  (class name → semantic mask ID, 0 = background)
# ============================================================================

PUMA_CLASS_MAP = {
    "apoptosis":    1,
    "tumor":        2,
    "endothelial":  3,
    "stroma":       4,
    "lymphocyte":   5,
    "histocyte":    6,
    "epithelial":   7,
    "melanophage":  8,
    "other":        9,
}

# CD8+ class ID (assigned separately via IoU matching, not from PUMA)
CD8_CLASS_ID = 10

# ============================================================================
# MISC
# ============================================================================

IS_WINDOWS = True            # Set to True on Windows
