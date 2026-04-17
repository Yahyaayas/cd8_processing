"""
Classpose PUMA Inference Wrapper
"""

import os
import shutil
import subprocess
import platform
from pathlib import Path
from typing import List, Tuple

import json
import numpy as np
from shapely.geometry import Polygon, shape
from shapely.affinity import affine_transform

from . import config
from .utils import load_geojson_polygons


def ensure_uv() -> str:
    """Install uv if not on PATH."""
    if shutil.which("uv"):
        return shutil.which("uv")

    print("[setup] Installing uv...")
    if config.IS_WINDOWS:
        subprocess.run(
            'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
            shell=True, check=True
        )
    else:
        subprocess.run(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            shell=True, check=True
        )

    # Update PATH
    for p in [Path.home() / ".local" / "bin", Path.home() / ".cargo" / "bin"]:
        if p.exists():
            os.environ["PATH"] = f"{p}{os.pathsep}{os.environ['PATH']}"

    return shutil.which("uv") or "uv"


def ensure_classpose():
    """Clone + setup Classpose. Idempotent."""
    config.CLASSPOSE_WORKDIR.mkdir(parents=True, exist_ok=True)
    config.CLASSPOSE_MODELS.mkdir(parents=True, exist_ok=True)

    ensure_uv()

    if not config.CLASSPOSE_REPO.exists():
        print(f"[setup] Cloning Classpose to {config.CLASSPOSE_REPO}")
        subprocess.run(
            f'git clone https://github.com/sohmandal/classpose.git "{config.CLASSPOSE_REPO}"',
            shell=True, check=True
        )

    marker = config.CLASSPOSE_REPO / ".venv" / "pyvenv.cfg"
    if not marker.exists():
        print("[setup] Running uv sync...")
        subprocess.run("uv sync", cwd=str(config.CLASSPOSE_REPO), shell=True, check=True)
        subprocess.run("uv pip install openslide-bin", cwd=str(config.CLASSPOSE_REPO),
                      shell=True, check=True)
    else:
        print("[setup] Classpose already installed")


def run_classpose_inference(
    slide_path: Path,
    output_folder: Path,
    model: str = config.CLASSPOSE_MODEL,
) -> Path:
    """
    Run Classpose inference on a slide.

    Returns:
        Path to output folder containing results
    """
    ensure_classpose()

    output_folder.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CLASSPOSE_MODEL_DIR"] = str(config.CLASSPOSE_MODELS)

    bf16 = "--bf16" if config.CLASSPOSE_BF16 else "--no-bf16"

    cmd = (
        f'uv run classpose-predict-wsi '
        f'--model_config {model} '
        f'--slide_path "{slide_path}" '
        f'--output_folder "{output_folder}" '
        f'--device {config.CLASSPOSE_DEVICE} '
        f'--batch_size {config.CLASSPOSE_BATCH_SIZE} '
        f'--tile_size {config.CLASSPOSE_TILE_SIZE} '
        f'--overlap {config.CLASSPOSE_OVERLAP} '
        f'{bf16}'
    )

    print(f"[Classpose] Running inference on {slide_path.name}...")
    subprocess.run(cmd, cwd=str(config.CLASSPOSE_REPO), env=env, shell=True, check=True)
    print(f"[Classpose] Done. Results in {output_folder}")

    return output_folder


def load_classpose_predictions(
    output_folder: Path,
) -> List[Tuple[Polygon, str]]:
    """
    Load ALL Classpose predictions from GeoJSON files (all PUMA classes).

    Returns:
        List of (polygon, class_name) tuples
    """
    candidates = sorted(output_folder.rglob("*_cell_contours.geojson")) \
                 or sorted(output_folder.rglob("*.geojson"))

    if not candidates:
        raise FileNotFoundError(f"No GeoJSON found in {output_folder}")

    polys = []
    for gj_path in candidates:
        for poly, name in load_geojson_polygons(gj_path):
            polys.append((poly, name or "other"))

    return polys


def run_classpose_on_he(
    he_slide_path: Path,
    output_folder: Path,
    force_rerun: bool = False,
) -> List[Tuple[Polygon, str]]:
    """
    Run Classpose on H&E slide and return lymphocyte predictions.

    Args:
        he_slide_path: Path to H&E SVS file
        output_folder: Where to save Classpose output
        force_rerun: Force re-running even if output exists

    Returns:
        List of (polygon, "lymphocyte") tuples in H&E level-0 coords
    """
    # Check if already run
    if not force_rerun and output_folder.exists():
        try:
            polys = load_classpose_predictions(output_folder)
            if polys:
                print(f"[Classpose] Reusing existing output: {len(polys)} predictions (all classes)")
                return polys
        except FileNotFoundError:
            pass

    # Run inference
    run_classpose_inference(he_slide_path, output_folder)

    # Load results
    polys = load_classpose_predictions(output_folder)
    print(f"[Classpose] Loaded {len(polys)} predictions (all PUMA classes)")

    return polys


def transform_polygons_to_ihc_space(
    polygons: List[Tuple[Polygon, str]],
    M_he2ihc: np.ndarray,
) -> List[Tuple[Polygon, str]]:
    """
    Transform H&E-space polygons to IHC coordinate space.

    Args:
        polygons: List of (polygon, class) in H&E space
        M_he2ihc: Affine transform H&E → IHC

    Returns:
        List of (polygon, class) in IHC space
    """
    # Convert cv2 affine to shapely affine params
    a, b, xoff = M_he2ihc[0]
    d, e, yoff = M_he2ihc[1]
    affine_params = [a, b, d, e, xoff, yoff]

    result = []
    for poly, cls in polygons:
        transformed = affine_transform(poly, affine_params)
        result.append((transformed, cls))

    return result


def transform_polygons_to_he_space(
    polygons: List[Tuple[Polygon, str]],
    M_ihc2he: np.ndarray,
) -> List[Tuple[Polygon, str]]:
    """
    Transform IHC-space polygons to H&E coordinate space.

    Args:
        polygons: List of (polygon, class) in IHC space
        M_ihc2he: Affine transform IHC → H&E

    Returns:
        List of (polygon, class) in H&E space
    """
    # Convert cv2 affine to shapely affine params
    a, b, xoff = M_ihc2he[0]
    d, e, yoff = M_ihc2he[1]
    affine_params = [a, b, d, e, xoff, yoff]

    result = []
    for poly, cls in polygons:
        transformed = affine_transform(poly, affine_params)
        result.append((transformed, cls))

    return result
