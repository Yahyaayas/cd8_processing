"""
CD8 Tile Processing Pipeline - Main Entry Point

Generates paired H&E/IHC tiles with CD8 masks from WSI data.

Usage:
    python -m CD8_processing
    python -m CD8_processing --case JRS-22-1351-A
    python -m CD8_processing --case JRS-22-1351-A --rerun-classpose
    python -m CD8_processing --unzip  # Unzip ZIP files first
"""

import argparse
import sys
from pathlib import Path

from . import config
from .utils import get_available_cases, ensure_dir, unzip_all_cases
from .tile_generator import process_single_case, save_summary, print_total_summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate paired H&E/IHC tiles with CD8 masks"
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Process specific case ID (e.g., JRS-22-1351-A). If not specified, process all cases."
    )
    parser.add_argument(
        "--rerun-classpose",
        action="store_true",
        help="Force re-run Classpose inference even if output exists"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: data/processed)"
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Unzip ZIP files in raw data directory before processing"
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default=None,
        help="Staging directory for unzipped files (default: data/raw/unzipped)"
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = config.PROCESSED_DIR

    ensure_dir(output_base)

    # Setup staging directory for unzipped files
    staging_dir = None
    if args.unzip or args.staging_dir:
        if args.staging_dir:
            staging_dir = Path(args.staging_dir)
        else:
            staging_dir = config.RAW_DATA_DIR / "unzipped"

    # Unzip if requested
    if args.unzip:
        print("\n[Unzip] Unzipping case files...")
        staging_dir = unzip_all_cases(config.RAW_DATA_DIR, staging_dir)
        print(f"[Unzip] Staging directory: {staging_dir}\n")

    # Get available cases
    cases = get_available_cases(config.RAW_DATA_DIR, config.IHC_LABEL_DIR, staging_dir)

    if not cases:
        print("No cases found!")
        print(f"  Raw data dir: {config.RAW_DATA_DIR}")
        print(f"  IHC label dir: {config.IHC_LABEL_DIR}")
        if staging_dir:
            print(f"  Staging dir: {staging_dir}")
        print("\nHint: Use --unzip to extract ZIP files first")
        sys.exit(1)

    # Filter by case if specified
    if args.case:
        cases = [c for c in cases if c["case_id"] == args.case]
        if not cases:
            print(f"Case '{args.case}' not found!")
            sys.exit(1)

    print(f"Found {len(cases)} case(s) to process:")
    for c in cases:
        print(f"  - {c['case_id']}")
    print()

    # Process each case
    summaries = []
    for case_info in cases:
        try:
            summary = process_single_case(
                case_info,
                output_base,
                force_rerun_classpose=args.rerun_classpose
            )
            summaries.append(summary)
        except Exception as e:
            print(f"ERROR processing {case_info['case_id']}: {e}")
            import traceback
            traceback.print_exc()

    # Save and print summary
    if summaries:
        summary_path = output_base / "processing_summary.json"
        save_summary(summaries, summary_path)
        print_total_summary(summaries)


if __name__ == "__main__":
    main()
