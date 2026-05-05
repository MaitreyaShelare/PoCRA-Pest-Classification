"""
Validate prepared dataset for completeness and correctness.

Usage:
    python scripts/validate_dataset.py --dataset data/processed
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict
import csv


def validate_dataset(dataset_dir: Path) -> Dict[str, any]:
    """
    Validate dataset structure and contents.
    
    Args:
        dataset_dir: Path to processed dataset
    
    Returns:
        Validation report dict
    """
    print(f"\nValidating dataset: {dataset_dir}\n")
    
    # Check for labels.csv
    labels_csv = dataset_dir / "labels.csv"
    if not labels_csv.exists():
        print(f"❌ ERROR: labels.csv not found at {labels_csv}")
        return None

    # Load labels.csv
    rows = []
    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"✓ Loaded {len(rows)} records from labels.csv")

    # Validate structure
    errors = []
    warnings = []

    # Check for missing files
    missing_files = []
    for row in rows:
        filepath = dataset_dir / row["filepath"]
        if not filepath.exists():
            missing_files.append(row["filepath"])

    if missing_files:
        errors.append(f"{len(missing_files)} image files referenced in CSV but not found on disk")
        for f in missing_files[:5]:
            print(f"  Missing: {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")

    # Check for orphaned files (images not in CSV)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = set()
    for ext in image_extensions:
        image_files.update(str(p.relative_to(dataset_dir)) for p in dataset_dir.rglob(f"*{ext}"))

    csv_files = set(row["filepath"] for row in rows)
    orphaned = image_files - csv_files
    if orphaned:
        warnings.append(f"{len(orphaned)} image files found but not in CSV")

    # Statistics
    print(f"\n✓ Statistics:")
    print(f"  Crops: {len(set(r['crop'] for r in rows))}")
    print(f"  Pests: {len(set(r['pest'] for r in rows))}")
    print(f"  Image types: {set(r['image_type'] for r in rows)}")

    # Split distribution
    splits = Counter(r["split"] for r in rows)
    print(f"\n✓ Split distribution:")
    for split, count in sorted(splits.items()):
        pct = 100 * count / len(rows)
        print(f"  {split:<10} {count:>6} ({pct:>5.1f}%)")

    # Per-crop stats
    print(f"\n✓ Per-crop sample counts:")
    crops = set(r["crop"] for r in rows)
    for crop in sorted(crops):
        crop_rows = [r for r in rows if r["crop"] == crop]
        print(f"  {crop:<15} {len(crop_rows):>6}")

    # Class imbalance check
    print(f"\n✓ Class imbalance check:")
    pest_counts = Counter(r["pest"] for r in rows)
    max_count = max(pest_counts.values())
    min_count = min(pest_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    print(f"  Max class: {max_count} samples")
    print(f"  Min class: {min_count} samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}x")
    
    if imbalance_ratio > 10:
        warnings.append(f"High class imbalance: {imbalance_ratio:.1f}x")

    # Results
    print(f"\n" + "=" * 70)
    if errors:
        print(f"❌ ERRORS ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"✓ All checks passed!")

    if warnings:
        print(f"\n⚠ WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")

    print("=" * 70 + "\n")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_records": len(rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate prepared dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/processed",
    )
    args = parser.parse_args()

    validate_dataset(args.dataset)