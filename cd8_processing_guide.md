# CD8 Tile Processing Pipeline - User Guide

## Overview

This pipeline generates paired H&E/IHC tiles with CD8 masks from whole-slide images (WSI). It combines:

1. **Global registration** (H&E ↔ IHC) using SIFT + RANSAC
2. **PUMA-Classpose inference** for lymphocyte detection on H&E
3. **IoU matching** to identify CD8+ nuclei (lymphocytes overlapping IHC positive cells)
4. **Tile generation** centered on CD8+ locations

---

## Folder Structure

```
Workspace/
├── data/
│   ├── raw/                           # Input: WSI files
│   │   ├── JRS-22-1351-A/
│   │   │   ├── HE_JRS-22-1351-A.svs
│   │   │   └── IHC_JRS-22-1351-A.svs
│   │   └── ...
│   │
│   ├── ihc_stain_label/               # Input: IHC GeoJSON labels
│   │   ├── IHC_JRS-22-1351-A.geojson
│   │   └── IHC_JRS-22-2585-B.geojson
│   │
│   └── processed/                     # Output: Generated tiles
│       ├── he_tiles/
│       ├── ihc_tiles/
│       ├── mask_tiles/
│       └── processing_summary.json
│
└── tools/CD8_processing/              # Pipeline code
    ├── config.py
    ├── utils.py
    ├── registration.py
    ├── classpose_wrapper.py
    ├── tile_generator.py
    └── main.py
```

---

## Quick Start

### 1. Prepare Data

Move IHC GeoJSON labels from raw folders to `ihc_stain_label/`:

```bash
# From Workspace root
cp data/raw/JRS-22-1351-A/IHC_*.geojson data/ihc_stain_label/
cp data/raw/JRS-22-2585-B/IHC_*.geojson data/ihc_stain_label/
# ... etc
```

### 2. Run Pipeline

```bash
cd D:\Users\aduhr\Documents\Internal Kuliah\File kuliah\Skripsi\Workspace

# If data is in ZIP format, unzip first
python -m CD8_processing --unzip

# Process all cases
python -m CD8_processing

# Process single case
python -m CD8_processing --case JRS-22-1351-A

# Force re-run Classpose (if needed)
python -m CD8_processing --case JRS-22-1351-A --rerun-classpose
```

---

## Configuration

Edit `tools/CD8_processing/config.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TILE_SIZE` | 1024 | Output tile size |
| `IOU_THRESHOLD` | 0.5 | Minimum IoU for CD8+ classification |
| `REG_LEVEL` | 2 | Pyramid level for registration |
| `CLASSPOSE_DEVICE` | "cuda:0" | GPU device |
| `CLASSPOSE_BATCH_SIZE` | 8 | Inference batch size |

---

## Pipeline Steps

```
┌─────────────────────────────────────────────┐
│ 1. LOAD DATA                                │
│    H&E SVS + IHC SVS + IHC GeoJSON         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 2. GLOBAL REGISTRATION                      │
│    SIFT features → RANSAC → Affine M        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 3. PUMA-CLASSPOSE INFERENCE                 │
│    Detect lymphocytes on H&E                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 4. IOU MATCHING                             │
│    PUMA ∩ IHC GeoJSON → CD8+ if IoU ≥ 0.5   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 5. TILE GENERATION                          │
│    Center 1024×1024 tiles on CD8+ nuclei    │
└─────────────────────────────────────────────┘
```

---

## Output Files

For each case, three folders are created:

| Folder | Contents |
|--------|----------|
| `he_tiles/{case}-he/` | H&E tiles (1024×1024 PNG) |
| `ihc_tiles/{case}-ihc/` | IHC tiles registered to H&E space |
| `mask_tiles/{case}-mask/` | CD8 instance masks (uint32 PNG) |

Each tile triplet shares the same filename: `tile_XXXX.png`

---

## Understanding the Mask

- **Value 0**: Background
- **Value 1-N**: CD8+ nuclei instance IDs
- Each nucleus has a unique ID for segmentation training

---

## Troubleshooting

### Classpose not found
First run auto-installs Classpose. Ensure `uv` is available or manually:
```bash
git clone https://github.com/sohmandal/classpose.git ~/classpose_workspace/classpose
cd ~/classpose_workspace/classpose
uv sync
uv pip install openslide-bin
```

### No cases found
Check that:
1. `data/raw/{case_id}/` contains both `HE_*.svs` and `IHC_*.svs`
2. `data/ihc_stain_label/` contains `IHC_{case_id}*.geojson`

**If data is in ZIP format:**
```bash
# Unzip all ZIP files to data/raw/unzipped/
python -m CD8_processing --unzip

# Or specify custom staging directory
python -m CD8_processing --unzip --staging-dir /path/to/staging
```

### Registration fails
- Ensure SVS files are not corrupted
- Try different `REG_LEVEL` in config.py

### Zero tiles generated
- Check IoU threshold (may be too strict)
- Verify IHC GeoJSON contains positive cells
- Check PUMA-Classpose output in `{case}/classpose_output/`

---

## Advanced Usage

### Custom Output Directory
```bash
python -m CD8_processing --output-dir /path/to/output
```

### Process Specific Cases
```bash
# Single case
python -m CD8_processing --case JRS-22-1351-A

# Multiple cases (modify main.py or run sequentially)
for case in JRS-22-1351-A JRS-22-2585-B; do
    python -m CD8_processing --case $case
done
```

---

## Summary JSON

After processing, `processing_summary.json` contains:
```json
[
  {
    "case_id": "JRS-22-1351-A",
    "registration_inliers": 1234,
    "puma_predictions": 5678,
    "ihc_positive_cells": 2345,
    "cd8_matched": 1234,
    "cd8_excluded": 444,
    "tiles_generated": 56
  }
]
```

---

## References

- **Notebook**: `experiments/test_imageprocessing_pipeline_single.ipynb`
- **Classpose**: https://github.com/sohmandal/classpose
- **Created**: 2026-04-15
