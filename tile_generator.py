"""
Tile Generator - Generate paired H&E/IHC tiles with CD8 masks
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import openslide
from PIL import Image
from shapely.geometry import Polygon

from . import config
from .utils import (
    normalize_case_id, ensure_dir, filter_patch,
    rasterize_polygons_to_mask, match_predictions_to_ground_truth
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
) -> Dict:
    """
    Process a single case: generate tiles with CD8+ nuclei.

    Args:
        case_info: Dict with case_id, he_path, ihc_path, geojson_path
        output_base: Base output directory
        force_rerun_classpose: Force re-running Classpose

    Returns:
        Summary dict with statistics
    """
    case_id = case_info["case_id"]
    norm_id = normalize_case_id(case_id)

    print(f"\n{'='*60}")
    print(f"Processing case: {case_id}")
    print(f"{'='*60}")

    # Create output directories
    he_out_dir = ensure_dir(output_base / "he_tiles" / f"{norm_id}-he")
    ihc_out_dir = ensure_dir(output_base / "ihc_tiles" / f"{norm_id}-ihc")
    mask_out_dir = ensure_dir(output_base / "mask_tiles" / f"{norm_id}-mask")

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

    print(f"\n[Step 4] Transform PUMA predictions to IHC space")
    puma_ihc = transform_polygons_to_ihc_space(
        [(p, c) for p, c in puma_predictions],
        M_he2ihc
    )
    puma_ihc_polys = [p for p, _ in puma_ihc]
    ihc_gt_polys = [p for p, _ in ihc_polygons]

    print(f"\n[Step 5] IoU matching (threshold = {config.IOU_THRESHOLD})")
    matched, excluded = match_predictions_to_ground_truth(
        puma_ihc_polys,
        ihc_gt_polys,
        config.IOU_THRESHOLD
    )
    print(f"  CD8+ (matched): {len(matched)}")
    print(f"  Excluded: {len(excluded)}")

    # Transform matched CD8+ polygons back to H&E space
    print(f"\n[Step 6] Transform CD8+ polygons to H&E space")
    cd8_he_polys = []
    for i, (poly, iou) in enumerate(matched):
        # Get original polygon from matched list
        orig_poly = puma_predictions[i][0]  # Original in H&E space
        cd8_he_polys.append((orig_poly, iou))

    print(f"  CD8+ nuclei in H&E space: {len(cd8_he_polys)}")

    print(f"\n[Step 7] Generate tiles centered on CD8+ nuclei")
    tiles_generated = generate_cd8_tiles(
        he_slide,
        ihc_slide,
        cd8_he_polys,
        M_ihc2he,
        he_out_dir,
        ihc_out_dir,
        mask_out_dir
    )

    summary = {
        "case_id": case_id,
        "registration_inliers": n_inliers,
        "puma_predictions": len(puma_predictions),
        "ihc_positive_cells": len(ihc_polygons),
        "cd8_matched": len(matched),
        "cd8_excluded": len(excluded),
        "tiles_generated": tiles_generated,
        "he_tiles_dir": str(he_out_dir),
        "ihc_tiles_dir": str(ihc_out_dir),
        "mask_tiles_dir": str(mask_out_dir),
    }

    print(f"\n{'='*60}")
    print(f"Summary for {case_id}:")
    print(f"  CD8+ nuclei: {len(matched)}")
    print(f"  Tiles generated: {tiles_generated}")
    print(f"{'='*60}\n")

    return summary


def generate_cd8_tiles(
    he_slide: openslide.OpenSlide,
    ihc_slide: openslide.OpenSlide,
    cd8_polygons: List[Tuple[Polygon, float]],
    M_ihc2he: np.ndarray,
    he_out_dir: Path,
    ihc_out_dir: Path,
    mask_out_dir: Path,
) -> int:
    """
    Generate 1024x1024 tiles centered on CD8+ nuclei.

    Args:
        he_slide: H&E slide
        ihc_slide: IHC slide
        cd8_polygons: List of (polygon, iou_score) in H&E space
        M_ihc2he: IHC → H&E transform
        he_out_dir: Output directory for H&E tiles
        ihc_out_dir: Output directory for IHC tiles
        mask_out_dir: Output directory for masks

    Returns:
        Number of tiles generated
    """
    tile_size = config.TILE_SIZE
    tile_idx = 0

    # Get slide dimensions
    he_w, he_h = he_slide.dimensions

    # Sort CD8 polygons by IoU (highest first) - process best matches first
    cd8_polygons_sorted = sorted(cd8_polygons, key=lambda x: x[1], reverse=True)

    # Track which nuclei have been covered
    covered_nuclei = set()

    for poly, iou_score in cd8_polygons_sorted:
        # Get polygon centroid
        centroid = poly.centroid
        cx, cy = centroid.x, centroid.y

        # Calculate tile origin (centered on nucleus)
        x0 = int(cx - tile_size // 2)
        y0 = int(cy - tile_size // 2)

        # Clamp to slide boundaries
        x0 = max(0, min(x0, he_w - tile_size))
        y0 = max(0, min(y0, he_h - tile_size))

        # Check if this tile already covers processed nuclei
        tile_key = (x0, y0)
        if tile_key in covered_nuclei:
            continue

        # Extract patches
        try:
            he_patch, ihc_patch = extract_registered_patch(
                he_slide, ihc_slide, x0, y0, tile_size, M_ihc2he
            )
        except Exception as e:
            print(f"  Warning: Could not extract tile at ({x0}, {y0}): {e}")
            continue

        # Filter H&E patch
        passed, reason = filter_patch(he_patch)
        if not passed:
            continue

        # Get polygons within this tile for mask
        tile_polys = []
        tile_box = Polygon([
            (x0, y0),
            (x0 + tile_size, y0),
            (x0 + tile_size, y0 + tile_size),
            (x0, y0 + tile_size)
        ])

        for p, _ in cd8_polygons:
            if tile_box.contains(p.centroid) or tile_box.intersects(p):
                # Clip polygon to tile
                clipped = p.intersection(tile_box)
                if not clipped.is_empty:
                    # Translate to tile-local coordinates (offset by x0, y0)
                    from shapely.affinity import translate
                    clipped = translate(clipped, xoff=-x0, yoff=-y0)

                    if clipped.geom_type == 'Polygon':
                        tile_polys.append(clipped)
                    elif clipped.geom_type == 'MultiPolygon':
                        tile_polys.extend(clipped.geoms)

        if not tile_polys:
            continue

        # Create mask
        mask = rasterize_polygons_to_mask(tile_polys, (tile_size, tile_size))

        # Save tiles
        tile_name = f"tile_{tile_idx:04d}.png"

        Image.fromarray(he_patch).save(he_out_dir / tile_name)
        Image.fromarray(ihc_patch).save(ihc_out_dir / tile_name)
        Image.fromarray(mask).save(mask_out_dir / tile_name)

        tile_idx += 1
        covered_nuclei.add(tile_key)

        if tile_idx % 10 == 0:
            print(f"  Generated {tile_idx} tiles...")

    return tile_idx


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

    print(f"Cases processed: {len(summaries)}")
    print(f"Total CD8+ nuclei: {total_cd8}")
    print(f"Total tiles generated: {total_tiles}")
    print(f"{'='*60}\n")
