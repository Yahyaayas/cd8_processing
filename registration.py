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
    Extract H&E patch and corresponding IHC patch (registered to H&E space).

    Args:
        he_slide: H&E slide object
        ihc_slide: IHC slide object
        x, y: Top-left corner in H&E level-0 coords
        size: Patch size
        M_ihc2he: Transform from IHC to H&E coords

    Returns:
        (he_patch, ihc_patch) - both in H&E coordinate space
    """
    # Extract H&E patch directly
    he_patch = np.array(he_slide.read_region((x, y), 0, (size, size)).convert("RGB"))

    # For IHC: transform H&E patch corners to IHC space
    corners_he = np.array([
        [x, y],
        [x + size, y],
        [x, y + size],
        [x + size, y + size],
    ], dtype=np.float32)

    # Use inverse of M_ihc2he to get H&E -> IHC
    M_he2ihc = cv2.invertAffineTransform(M_ihc2he)
    corners_ihc = cv2.transform(corners_he.reshape(-1, 1, 2), M_he2ihc).reshape(-1, 2)

    # Get bounding box in IHC space
    x0_ihc = int(np.floor(corners_ihc[:, 0].min()))
    y0_ihc = int(np.floor(corners_ihc[:, 1].min()))
    x1_ihc = int(np.ceil(corners_ihc[:, 0].max()))
    y1_ihc = int(np.ceil(corners_ihc[:, 1].max()))

    # Clip to IHC slide dimensions
    iw, ih = ihc_slide.dimensions
    x0_ihc = max(0, x0_ihc)
    y0_ihc = max(0, y0_ihc)
    x1_ihc = min(iw, x1_ihc)
    y1_ihc = min(ih, y1_ihc)

    w_ihc = max(x1_ihc - x0_ihc, 1)
    h_ihc = max(y1_ihc - y0_ihc, 1)

    # Read IHC region
    ihc_region = np.array(ihc_slide.read_region((x0_ihc, y0_ihc), 0, (w_ihc, h_ihc)).convert("RGB"))

    # Create local transform to warp IHC region to H&E patch space
    M_local = M_he2ihc.copy()
    M_local[0, 2] -= x0_ihc
    M_local[1, 2] -= y0_ihc

    # Warp IHC to match H&E patch
    ihc_patch = cv2.warpAffine(ihc_region, M_local, (size, size),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

    return he_patch, ihc_patch
