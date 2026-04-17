"""
H&E ↔ IHC Registration Module
"""

import numpy as np
import cv2
import openslide
from typing import Tuple

from . import config


def detect_and_match_sift(
    img1: np.ndarray,
    img2: np.ndarray,
    n_features: int = config.SIFT_NFEATURES,
    ratio: float = config.LOWE_RATIO,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Detect and match SIFT features between two images.

    Returns:
        (pts1, pts2, n_matches) - matched keypoints from img1 and img2
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create(nfeatures=n_features)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < ratio * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return pts1, pts2, len(good)


def estimate_affine_transform(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    threshold: float = config.RANSAC_THRESHOLD,
    confidence: float = config.RANSAC_CONFIDENCE,
    max_iters: int = config.RANSAC_MAX_ITERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate affine transform using RANSAC.

    Returns:
        (M, inliers_mask) - transformation matrix and inlier mask
    """
    M, inliers_mask = cv2.estimateAffinePartial2D(
        pts_src, pts_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        confidence=confidence,
        maxIters=max_iters,
    )
    return M, inliers_mask


def register_he_to_ihc(
    he_slide: openslide.OpenSlide,
    ihc_slide: openslide.OpenSlide,
    level: int = config.REG_LEVEL,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute affine transform from H&E to IHC coordinate space.

    Returns:
        (M_he2ihc, M_ihc2he, n_inliers) - forward and inverse transforms
    """
    # Load low-res thumbnails
    he_low = load_wsi_at_level_path(he_slide, level)
    ihc_low = load_wsi_at_level_path(ihc_slide, level)

    # Get downsamples
    he_ds = he_slide.level_downsamples[level]
    ihc_ds = ihc_slide.level_downsamples[level]

    # Match features
    pts_he, pts_ihc, n_matches = detect_and_match_sift(he_low, ihc_low)

    # Estimate transform at low resolution
    M_low, inliers_mask = estimate_affine_transform(pts_he, pts_ihc)
    n_inliers = int(inliers_mask.sum())

    # Rescale to level-0
    M_he2ihc = M_low.copy()
    M_he2ihc[:, :2] = M_he2ihc[:, :2] * (ihc_ds / he_ds)
    M_he2ihc[:, 2] = M_he2ihc[:, 2] * ihc_ds

    # Inverse transform
    M_ihc2he = cv2.invertAffineTransform(M_he2ihc)

    return M_he2ihc, M_ihc2he, n_inliers


def load_wsi_at_level_path(slide: openslide.OpenSlide, level: int) -> np.ndarray:
    """Load WSI at given level as numpy array."""
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    return np.array(img)


def transform_polygon_coords(
    points: np.ndarray,
    M: np.ndarray
) -> np.ndarray:
    """
    Transform polygon coordinates using affine matrix.

    Args:
        points: (N, 2) array of points
        M: (2, 3) affine transformation matrix

    Returns:
        Transformed points
    """
    points_homog = np.column_stack([points, np.ones(len(points))])
    transformed = (M @ points_homog.T).T
    return transformed[:, :2]


def extract_registered_patch(
    he_slide: openslide.OpenSlide,
    ihc_slide: openslide.OpenSlide,
    x: int,
    y: int,
    size: int,
    M_ihc2he: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract H&E patch and corresponding IHC patch at registered coordinates.

    IHC is NOT warped — instead the tile top-left (x, y) is transformed into
    IHC space and the IHC patch is read directly there. This avoids blank tiles
    from incorrect warp lookups outside the IHC region.

    Args:
        he_slide: H&E slide object
        ihc_slide: IHC slide object
        x, y: Top-left corner in H&E level-0 coords
        size: Patch size
        M_ihc2he: Transform from IHC to H&E coords (inverted to get H&E→IHC)

    Returns:
        (he_patch, ihc_patch) as RGB numpy arrays
    """
    # Extract H&E patch directly
    he_patch = np.array(he_slide.read_region((x, y), 0, (size, size)).convert("RGB"))

    # Transform H&E top-left corner to IHC space
    M_he2ihc = cv2.invertAffineTransform(M_ihc2he)
    pt_he = np.array([[[x, y]]], dtype=np.float32)
    pt_ihc = cv2.transform(pt_he, M_he2ihc).reshape(2)
    x0_ihc = int(round(pt_ihc[0]))
    y0_ihc = int(round(pt_ihc[1]))

    # Clamp to IHC slide boundaries
    iw, ih = ihc_slide.dimensions
    x0_ihc = max(0, min(x0_ihc, iw - size))
    y0_ihc = max(0, min(y0_ihc, ih - size))

    # Read IHC patch at registered position — same tile size, no warping
    ihc_patch = np.array(ihc_slide.read_region((x0_ihc, y0_ihc), 0, (size, size)).convert("RGB"))

    return he_patch, ihc_patch
