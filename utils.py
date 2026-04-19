"""
Utility functions for CD8 Tile Processing Pipeline
"""

import json
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import openslide
from shapely.geometry import Polygon, shape
from shapely.strtree import STRtree

from . import config


def load_wsi_at_level(slide_path: Path, level: int = 0) -> np.ndarray:
    """Load WSI at specified pyramid level as RGB numpy array."""
    slide = openslide.OpenSlide(str(slide_path))
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    return np.array(img)


def load_geojson_polygons(geojson_path: Path) -> List[Tuple[Polygon, Optional[str]]]:
    """
    Load QuPath-style GeoJSON → list of (shapely.Polygon, class_name).

    Args:
        geojson_path: Path to GeoJSON file

    Returns:
        List of (polygon, class_name) tuples
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    feats = gj["features"] if gj.get("type") == "FeatureCollection" else gj
    out = []

    for feat in feats:
        geom = shape(feat["geometry"])
        cls = feat.get("properties", {}).get("classification", {})
        name = cls.get("name") if isinstance(cls, dict) else None

        if geom.is_empty:
            continue

        if geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                out.append((g, name))
        elif geom.geom_type == "Polygon":
            out.append((geom, name))

    return out


def filter_patch(patch: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Filter a single RGB patch by quality metrics.

    Returns:
        (passed: bool, reason: str | None)
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    white_ratio = np.sum(gray > 150) / gray.size
    black_ratio = np.sum(gray < 30) / gray.size
    std_dev = float(np.std(gray))
    rgb_mean = float(np.mean(patch))

    if rgb_mean < config.RGB_MEAN_MIN:
        return False, "low_rgb"
    if black_ratio > config.BLACK_THRESHOLD:
        return False, "black"
    if white_ratio > config.BACKGROUND_THRESHOLD:
        return False, "white"
    if std_dev < config.GREY_THRESHOLD:
        return False, "grey"

    return True, None


def polygon_to_cv2_contour(poly: Polygon) -> np.ndarray:
    """Convert shapely Polygon to OpenCV contour format."""
    coords = np.array(poly.exterior.coords.xy, dtype=np.float32).T
    return coords.reshape(-1, 1, 2)


def rasterize_polygons_to_mask(
    polygons: List[Polygon],
    mask_shape: Tuple[int, int],
    instance_ids: List[int] = None,
) -> np.ndarray:
    """
    Rasterize polygons to instance mask using OpenCV.

    Args:
        polygons: List of shapely Polygons
        mask_shape: (height, width) of output mask
        instance_ids: Instance ID for each polygon (default: 1..N)

    Returns:
        uint32 instance mask
    """
    h, w = mask_shape
    mask = np.zeros((h, w), dtype=np.uint32)

    if instance_ids is None:
        instance_ids = list(range(1, len(polygons) + 1))

    for poly, inst_id in zip(polygons, instance_ids):
        # Get bounding box
        minx, miny, maxx, maxy = poly.bounds
        x0, y0 = int(max(0, minx)), int(max(0, miny))
        x1, y1 = int(min(w, maxx)), int(min(h, maxy))

        if x1 <= x0 or y1 <= y0:
            continue

        # Create temporary mask for this polygon
        temp_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        contour = polygon_to_cv2_contour(poly)
        contour_offset = contour - np.array([x0, y0])

        cv2.fillPoly(temp_mask, [contour_offset.astype(np.int32)], 255)

        # Set instance ID
        mask[y0:y1, x0:x1][temp_mask > 0] = inst_id

    return mask


def rasterize_polygons_to_semantic_mask(
    polygons_with_classes: List[Tuple[Polygon, int]],
    mask_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Rasterize polygons to a semantic class mask.

    Args:
        polygons_with_classes: List of (polygon, class_id) in tile-local coords
        mask_shape: (height, width) of output mask

    Returns:
        uint8 semantic mask with class IDs (0=background, 1-11=PUMA classes)
    """
    h, w = mask_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for poly, class_id in polygons_with_classes:
        minx, miny, maxx, maxy = poly.bounds
        x0, y0 = int(max(0, minx)), int(max(0, miny))
        x1, y1 = int(min(w, maxx)), int(min(h, maxy))

        if x1 <= x0 or y1 <= y0:
            continue

        temp_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        contour = polygon_to_cv2_contour(poly)
        contour_offset = contour - np.array([x0, y0])
        cv2.fillPoly(temp_mask, [contour_offset.astype(np.int32)], 255)
        mask[y0:y1, x0:x1][temp_mask > 0] = class_id

    return mask


def calculate_iou(poly1: Polygon, poly2: Polygon) -> float:
    """Calculate IoU between two polygons."""
    if not poly1.intersects(poly2):
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_ground_truth(
    pred_polys: List[Polygon],
    gt_polys: List[Polygon],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[Polygon, float]], List[Tuple[Polygon, float]]]:
    """
    Match predictions to ground truth using IoU.

    Returns:
        (matched, unmatched) where each is list of (polygon, iou_score)
    """
    if not gt_polys:
        return [], [(p, 0.0) for p in pred_polys]

    tree = STRtree(gt_polys)
    matched, unmatched = [], []

    for pred_poly in pred_polys:
        candidate_idxs = tree.query(pred_poly)
        best_iou = 0.0

        for idx in candidate_idxs:
            score = calculate_iou(pred_poly, gt_polys[int(idx)])
            if score > best_iou:
                best_iou = score

        if best_iou >= iou_threshold:
            matched.append((pred_poly, best_iou))
        else:
            unmatched.append((pred_poly, best_iou))

    return matched, unmatched


def get_available_cases(raw_dir: Path, ihc_label_dir: Path) -> List[dict]:
    """
    Get available cases (have both H&E, IHC SVS and IHC GeoJSON).

    Returns:
        List of dicts with case info: {case_id, he_path, ihc_path, geojson_path}
    """
    cases = []

    for case_folder in raw_dir.iterdir():
        if not case_folder.is_dir():
            continue

        case_id = case_folder.name

        # Find H&E SVS
        he_path = None
        for f in case_folder.glob("HE_*.svs"):
            he_path = f
            break

        # Find IHC SVS
        ihc_path = None
        for f in case_folder.glob("IHC_*.svs"):
            ihc_path = f
            break

        # Find IHC GeoJSON (in ihc_stain_label folder)
        geojson_path = None
        for f in ihc_label_dir.glob(f"IHC_{case_id}*.geojson"):
            geojson_path = f
            break

        if he_path and ihc_path and geojson_path:
            cases.append({
                "case_id": case_id,
                "he_path": he_path,
                "ihc_path": ihc_path,
                "geojson_path": geojson_path,
            })

    return cases


def normalize_case_id(case_id: str) -> str:
    """Normalize case ID to lowercase for folder names."""
    return case_id.lower().replace("_", "-")


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_zip_files(raw_dir: Path) -> List[Path]:
    """
    Find all ZIP files in raw data directory.

    Returns:
        List of ZIP file paths
    """
    return list(raw_dir.glob("*.zip")) + list(raw_dir.glob("*/*.zip"))


def unzip_case_file(zip_path: Path, output_dir: Path) -> Path:
    """
    Unzip a case ZIP file to output directory.

    Args:
        zip_path: Path to ZIP file
        output_dir: Directory to extract to

    Returns:
        Path to extracted case folder
    """
    ensure_dir(output_dir)

    # Get case ID from zip filename (e.g., JRS-22-1351-A.zip -> JRS-22-1351-A)
    case_id = zip_path.stem
    extract_dir = output_dir / case_id

    # Skip if already extracted
    if extract_dir.exists():
        print(f"[Unzip] Already extracted: {case_id}")
        return extract_dir

    print(f"[Unzip] Extracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"[Unzip] Extracted to: {extract_dir}")
    return extract_dir


def unzip_all_cases(raw_dir: Path, staging_dir: Path = None) -> Path:
    """
    Unzip all case ZIP files found in raw directory.

    Args:
        raw_dir: Directory containing ZIP files
        staging_dir: Where to extract (default: raw_dir / "unzipped")

    Returns:
        Path to staging directory with extracted files
    """
    if staging_dir is None:
        staging_dir = raw_dir / "unzipped"

    zip_files = find_zip_files(raw_dir)

    if not zip_files:
        print(f"[Unzip] No ZIP files found in {raw_dir}")
        return raw_dir

    print(f"[Unzip] Found {len(zip_files)} ZIP file(s)")

    for zip_path in zip_files:
        unzip_case_file(zip_path, staging_dir)

    return staging_dir


def extract_case_id(filename: str) -> str:
    """
    Extract case ID from filename like 'HE_JRS-22-1351-A.svs' -> 'JRS-22-1351-A'.

    Supports formats:
    - HE_JRS-22-1351-A.svs -> JRS-22-1351-A
    - IHC_JRS-22-1351-A.svs -> JRS-22-1351-A
    - HE_JRS-22-1351-A.svs.zip -> JRS-22-1351-A
    """
    # Remove extension
    name = Path(filename).stem

    # Remove HE_ or IHC_ prefix
    for prefix in ["HE_", "IHC_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    return name


def get_available_cases(
    raw_dir: Path,
    ihc_label_dir: Path,
    staging_dir: Path = None
) -> List[dict]:
    """
    Get available cases (have both H&E, IHC SVS and IHC GeoJSON).

    Supports both nested structure (case folders) and flat structure (files directly in raw_dir).

    Args:
        raw_dir: Directory with raw data (may contain ZIP files)
        ihc_label_dir: Directory with IHC GeoJSON labels
        staging_dir: Staging directory for unzipped files

    Returns:
        List of dicts with case info: {case_id, he_path, ihc_path, geojson_path}
    """
    cases = []

    # Directories to search: raw_dir and optionally staging_dir
    search_dirs = [raw_dir]
    if staging_dir and staging_dir.exists():
        search_dirs.append(staging_dir)

    for search_dir in search_dirs:
        # Check for case folders (nested structure)
        for case_folder in search_dir.iterdir():
            if not case_folder.is_dir():
                continue

            # Skip staging directory itself
            if case_folder.name == "unzipped":
                continue

            case_id = case_folder.name

            # Skip if already processed
            if any(c["case_id"] == case_id for c in cases):
                continue

            # Find H&E SVS
            he_path = None
            for f in case_folder.glob("HE_*.svs"):
                he_path = f
                break

            # Find IHC SVS
            ihc_path = None
            for f in case_folder.glob("IHC_*.svs"):
                ihc_path = f
                break

            # Find IHC GeoJSON (in ihc_stain_label folder)
            geojson_path = None
            for f in ihc_label_dir.glob(f"IHC_{case_id}*.geojson"):
                geojson_path = f
                break

            if he_path and ihc_path and geojson_path:
                cases.append({
                    "case_id": case_id,
                    "he_path": he_path,
                    "ihc_path": ihc_path,
                    "geojson_path": geojson_path,
                })

    # Also check for flat structure (files directly in raw_dir)
    if not cases and raw_dir.is_dir():
        # Find all HE_*.svs files
        he_files = list(raw_dir.glob("HE_*.svs"))

        for he_path in he_files:
            case_id = extract_case_id(he_path.name)

            # Skip if already processed
            if any(c["case_id"] == case_id for c in cases):
                continue

            # Find matching IHC file
            ihc_pattern = f"IHC_{case_id}.svs"
            ihc_path = raw_dir / ihc_pattern
            if not ihc_path.exists():
                # Try case-insensitive
                ihc_path = None
                for f in raw_dir.glob("IHC_*.svs"):
                    if extract_case_id(f.name) == case_id:
                        ihc_path = f
                        break

            if not ihc_path:
                continue

            # Find IHC GeoJSON
            geojson_path = None
            for f in ihc_label_dir.glob(f"IHC_{case_id}*.geojson"):
                geojson_path = f
                break
            if not geojson_path:
                # Try case-insensitive
                for f in ihc_label_dir.glob("IHC_*.geojson"):
                    if extract_case_id(f.name) == case_id:
                        geojson_path = f
                        break

            if geojson_path:
                cases.append({
                    "case_id": case_id,
                    "he_path": he_path,
                    "ihc_path": ihc_path,
                    "geojson_path": geojson_path,
                })

    return cases
