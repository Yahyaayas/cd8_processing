"""
Configuration for CD8 Tile Processing Pipeline
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Workspace root
WORKSPACE = Path(r"D:\Users\aduhr\Documents\Internal Kuliah\File kuliah\Skripsi\Workspace")

# Input paths
DATA_ROOT = WORKSPACE / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
IHC_LABEL_DIR = DATA_ROOT / "ihc_stain_label"

# Output paths
PROCESSED_DIR = DATA_ROOT / "processed"
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

LYMPHOCYTE_CLASS = "lymphocyte"    # Class name to filter from PUMA

# ============================================================================
# MISC
# ============================================================================

IS_WINDOWS = True            # Set to True on Windows
