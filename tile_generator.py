"""
Tile Generator - Random tile sampling from WSI with CD8+ labeling
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import openslide
from PIL import Image
from shapely.geometry import Polygon
from shapely.affinity import translate as shp_translate

from . import config
from .utils import (
    normalize_case_id, ensure_dir, filter_patch,
    rasterize_polygons_to_mask, rasterize_polygons_to_semantic_mask,
    match_predictions_to_ground_truth
)
from .registration import register_he_to_ihc, extract_registered_patch
from .classpose_wrapper import (
    run_classpose_on_he,
    transform_polygons_to_ihc_space,
    transform_polygons_to_he_space
)
from .utils import load_geojson_polygons


def process_single_case(
    case_info: Dict,
    output_base: Path,
    force_rerun_classpose: bool = False,
    cd8_class_id: int = None,
) -> Dict:
    """
    Process a single case: random tile sampling from WSI with CD8+ labeling.

    Args:
        case_info: Dict with case_id, he_path, ihc_path, geojson_path
        output_base: Base output directory
        force_rerun_classpose: Force re-running Classpose

    Returns:
        Summary dict with statistics
    """
    case_id = case_info["case_id"]
    norm_id = normalize_case_id(case_id)

    if cd8_class_id is None:
        cd8_class_id = config.CD8_CLASS_ID

    print(f"\n{'='*60}")
    print(f"Processing case: {case_id}")
    print(f"{'='*60}")

    # Create output directories
    he_out_dir = ensure_dir(output_base / "he_tiles" / f"{norm_id}-he")
    ihc_out_dir = ensure_dir(output_base / "ihc_tiles" / f"{norm_id}-ihc")
    mask_out_dir = ensure_dir(output_base / "mask_tiles" / f"{norm_id}-mask")

    existing_tiles = list(he_out_dir.glob("tile_*.png"))
    if existing_tiles:
        print(f"  Found {len(existing_tiles)} existing tiles — skipping case.")
        return {
            "case_id": case_id,
            "registration_inliers": 0,
            "puma_predictions": 0,
            "ihc_positive_cells": 0,
            "cd8_matched": 0,
            "cd8_excluded": 0,
            "tiles_generated": len(existing_tiles),
            "cd8_tiles": 0,
            "non_cd8_tiles": 0,
            "he_tiles_dir": str(he_out_dir),
            "ihc_tiles_dir": str(ihc_out_dir),
            "mask_tiles_dir": str(mask_out_dir),
            "skipped": True,
        }

    # Open slides
    he_slide = openslide.OpenSlide(str(case_info["he_path"]))
    ihc_slide = openslide.OpenSlide(str(case_info["ihc_path"]))

    print(f"[Step 1] Registration: H&E → IHC")
    M_he2ihc, M_ihc2he, n_inliers = register_he_to_ihc(he_slide, ihc_slide)
    print(f"  Inliers: {n_inliers}")

    print(f"\n[Step 2] PUMA-Classpose inference on H&E")
    classpose_out = ensure_dir(case_info["he_path"].parent / "classpose_output")
    puma_predictions = run_classpose_on_he(
        case_info["he_path"],
        classpose_out,
        force_rerun=force_rerun_classpose
    )

    print(f"\n[Step 3] Load IHC GeoJSON labels")
    ihc_polygons = load_geojson_polygons(case_info["geojson_path"])
    print(f"  IHC positive cells: {len(ihc_polygons)}")

    # Filter only lymphocytes from PUMA predictions for CD8 matching
    lymphocyte_predictions = [
        (p, c) for p, c in puma_predictions
        if c and c.lower() == "lymphocyte"
    ]
    print(f"  PUMA lymphocytes: {len(lymphocyte_predictions)}")

    print(f"\n[Step 4] Transform lymphocyte predictions to IHC space")
    lymph_ihc = transform_polygons_to_ihc_space(
        lymphocyte_predictions,
        M_he2ihc
    )
    lymph_ihc_polys = [(p_ihc, p_he) for (p_ihc, _), (p_he, _) in zip(lymph_ihc, lymphocyte_predictions)]
    ihc_gt_polys = [p for p, _ in ihc_polygons]

    print(f"\n[Step 5] IoU matching — lymphocytes vs IHC labels (threshold = {config.IOU_THRESHOLD})")
    matched, excluded = match_predictions_to_ground_truth(
        [p for p, _ in lymph_ihc_polys],
        ihc_gt_polys,
        config.IOU_THRESHOLD
    )
    print(f"  CD8+ (matched): {len(matched)}")
    print(f"  Excluded: {len(excluded)}")

    # Build lookup: IHC-space polygon → original H&E polygon
    ihc_to_he = {p_ihc.wkt: p_he for p_ihc, p_he in lymph_ihc_polys}

    # Get CD8+ polygons back in H&E space
    print(f"\n[Step 6] Collect CD8+ polygons in H&E space")
    cd8_he_polys = []
    for ihc_poly, iou in matched:
        he_orig = ihc_to_he.get(ihc_poly.wkt)
        if he_orig is None:
            continue
        cd8_he_polys.append((he_orig, iou))

    print(f"  CD8+ nuclei in H&E space: {len(cd8_he_polys)}")

    print(f"\n[Step 7] Random tile sampling from WSI (max {config.MAX_TILE} tiles)")
    tiles_generated, cd8_tiles, non_cd8_tiles = generate_random_tiles(
        he_slide,
        ihc_slide,
        cd8_he_polys,
        puma_predictions,
        M_ihc2he,
        he_out_dir,
        ihc_out_dir,
        mask_out_dir,
        cd8_class_id
    )

    summary = {
        "case_id": case_id,
        "registration_inliers": n_inliers,
        "puma_predictions": len(puma_predictions),
        "ihc_positive_cells": len(ihc_polygons),
        "cd8_matched": len(matched),
        "cd8_excluded": len(excluded),
        "tiles_generated": tiles_generated,
        "cd8_tiles": cd8_tiles,
        "non_cd8_tiles": non_cd8_tiles,
        "he_tiles_dir": str(he_out_dir),
        "ihc_tiles_dir": str(ihc_out_dir),
        "mask_tiles_dir": str(mask_out_dir),
    }

    print(f"\n{'='*60}")
    print(f"Summary for {case_id}:")
    print(f"  CD8+ nuclei: {len(matched)}")
    print(f"  Tiles: {tiles_generated} (CD8+: {cd8_tiles}, non-CD8+: {non_cd8_tiles})")
    print(f"{'='*60}\n")

    return summary


def generate_random_tiles(
    he_slide: openslide.OpenSlide,
    ihc_slide: openslide.OpenSlide,
    cd8_polygons: List[Tuple[Polygon, float]],
    puma_all_predictions: List[Tuple[Polygon, str]],
    M_ihc2he: np.ndarray,
    he_out_dir: Path,
    ihc_out_dir: Path,
    mask_out_dir: Path,
    cd8_class_id: int = None,
) -> Tuple[int, int, int]:
    """
    Random tile sampling from WSI with CD8+ labeling via IHC verification.
    Saves ALL tiles regardless of CD8+ content.

    Args:
        he_slide: H&E slide
        ihc_slide: IHC slide
        cd8_polygons: List of (polygon, iou_score) in H&E space (for CD8+ class assignment only)
        puma_all_predictions: All Classpose predictions (polygon, class_name) in H&E space
        M_ihc2he: IHC → H&E transform
        he_out_dir: Output directory for H&E tiles
        ihc_out_dir: Output directory for IHC tiles
        mask_out_dir: Output directory for masks (semantic PUMA class labels)

    Returns:
        (total_tiles, cd8_tile_count, non_cd8_tile_count)
    """
    random.seed(42)

    tile_size = config.TILE_SIZE
    max_tile = config.MAX_TILE

    tile_idx = 0
    cd8_tile_count = 0
    non_cd8_tile_count = 0

    if cd8_class_id is None:
        cd8_class_id = config.CD8_CLASS_ID

    he_w, he_h = he_slide.dimensions
    used_positions = set()
    max_attempts = max_tile * 5

    while tile_idx < max_tile and len(used_positions) < max_attempts:
        x0 = random.randint(0, max(0, he_w - tile_size - 1))
        y0 = random.randint(0, max(0, he_h - tile_size - 1))
        tile_key = (x0, y0)

        if tile_key in used_positions:
            continue
        used_positions.add(tile_key)

        try:
            he_patch, ihc_patch = extract_registered_patch(
                he_slide, ihc_slide, x0, y0, tile_size, M_ihc2he
            )
        except Exception as e:
            continue

        passed, reason = filter_patch(he_patch)
        if not passed:
            continue

        tile_box = Polygon([
            (x0, y0),
            (x0 + tile_size, y0),
            (x0 + tile_size, y0 + tile_size),
            (x0, y0 + tile_size)
        ])

        tile_polys_with_class = []
        has_cd8 = False

        for p, class_name in puma_all_predictions:
            if not (tile_box.contains(p.centroid) or tile_box.intersects(p)):
                continue
            clipped = p.intersection(tile_box)
            if clipped.is_empty:
                continue
            clipped = shp_translate(clipped, xoff=-x0, yoff=-y0)

            is_cd8 = any(p.equals(cd8_poly) for cd8_poly, _ in cd8_polygons)

            if is_cd8:
                class_id = cd8_class_id
                has_cd8 = True
            else:
                class_id = config.PUMA_CLASS_MAP.get(
                    class_name.lower().replace(" ", "_"), 11
                )

            geoms = clipped.geoms if clipped.geom_type == 'MultiPolygon' else [clipped]
            for g in geoms:
                if g.geom_type == 'Polygon':
                    tile_polys_with_class.append((g, class_id))

        if not tile_polys_with_class:
            continue

        mask = rasterize_polygons_to_semantic_mask(
            tile_polys_with_class, (tile_size, tile_size)
        )

        tile_name = f"tile_{tile_idx:04d}.png"
        Image.fromarray(he_patch).save(he_out_dir / tile_name)
        Image.fromarray(ihc_patch).save(ihc_out_dir / tile_name)
        Image.fromarray(mask).save(mask_out_dir / tile_name)

        tile_idx += 1
        if has_cd8:
            cd8_tile_count += 1
        else:
            non_cd8_tile_count += 1

        if tile_idx % 50 == 0:
            print(
                f"  {tile_idx}/{max_tile} tiles "
                f"(CD8+: {cd8_tile_count}, non-CD8+: {non_cd8_tile_count})"
            )

    print(
        f"  Done: {tile_idx} tiles "
        f"(CD8+: {cd8_tile_count}, non-CD8+: {non_cd8_tile_count})"
    )
    return tile_idx, cd8_tile_count, non_cd8_tile_count


def save_summary(summaries: List[Dict], output_path: Path):
    """Save processing summary to JSON."""
    with open(output_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSummary saved to: {output_path}")


def print_total_summary(summaries: List[Dict]):
    """Print overall summary across all cases."""
    print(f"\n{'='*60}")
    print("TOTAL SUMMARY")
    print(f"{'='*60}")

    total_cd8 = sum(s["cd8_matched"] for s in summaries)
    total_tiles = sum(s["tiles_generated"] for s in summaries)
    total_cd8_tiles = sum(s.get("cd8_tiles", 0) for s in summaries)
    total_non_cd8_tiles = sum(s.get("non_cd8_tiles", 0) for s in summaries)

    print(f"Cases processed: {len(summaries)}")
    print(f"Total CD8+ nuclei: {total_cd8}")
    print(f"Total tiles: {total_tiles}")
    print(f"  CD8+ tiles: {total_cd8_tiles}")
    print(f"  Non-CD8+ tiles: {total_non_cd8_tiles}")
    if total_tiles > 0:
        print(f"  % CD8+: {100 * total_cd8_tiles / total_tiles:.1f}%")
    print(f"{'='*60}\n")
