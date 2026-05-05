# # Auto-detect without prompts (recommended for batch processing)
# python scripts/prepare_dataset.py --raw data/raw --processed data/processed

# # Interactive: review and confirm auto-detected types
# python scripts/prepare_dataset.py --raw data/raw --processed data/processed --interactive

# # Force all merged to symptom (safe default)
# python scripts/prepare_dataset.py --raw data/raw --processed data/processed --default symptom

# # Force all merged to pest_body
# python scripts/prepare_dataset.py --raw data/raw --processed data/processed --default pest_body

# # Preview without copying
# python scripts/prepare_dataset.py --raw data/raw --processed data/processed --dry-run


"""
Complete dataset preparation pipeline.
Converts raw agricultural pest dataset to training-ready format.

UPDATED: Automatically detects predominant type (pest_body vs symptom) 
for each merged class based on visual heuristics, with option to override.

Raw dataset structure (from user):
    data/raw/
    ├── Brinjal/
    │   ├── Aphid/
    │   ├── Healthy/
    │   └── ...
    ├── Cotton/
    │   ├── Aphid/
    │   ├── Leaf_Roller/
    │   │   ├── Pest/
    │   │   ├── Symptoms/
    │   └── ...
    └── ...

Output structure (training-ready):
    data/processed/
    ├── Cotton/
    │   ├── Aphid/
    │   │   ├── pest_body/
    │   │   └── symptom/
    │   └── ...
    
    labels.csv (with columns: filepath, crop, pest, image_type, species_id, split)

Usage:
    # Auto-detect with smart defaults (recommended)
    python scripts/prepare_dataset.py --raw data/raw --processed data/processed

    # Preview with auto-detected defaults
    python scripts/prepare_dataset.py --raw data/raw --processed data/processed --dry-run

    # Interactive mode: review auto-detected, override if needed
    python scripts/prepare_dataset.py --raw data/raw --processed data/processed --interactive

    # Force all merged to symptom (safe default for imbalanced data)
    python scripts/prepare_dataset.py --raw data/raw --processed data/processed --default symptom

    # Force all merged to pest_body
    python scripts/prepare_dataset.py --raw data/raw --processed data/processed --default pest_body

    # Custom train/val/test split
    python scripts/prepare_dataset.py --raw data/raw --processed data/processed \
        --train-split 0.7 --val-split 0.15 --test-split 0.15
"""

import argparse
import csv
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional
import sys
from PIL import Image
import numpy as np


# ── Canonical name mapping ───────────────────────────────────────────────────

CROP_NAMES = {
    "cotton": "Cotton",
    "sorghum": "Sorghum",
    "chickpea": "Chickpea",
    "mango": "Mango",
    "tomato": "Tomato",
    "mustard": "Mustard",
    "maize": "Maize",
    "brinjal": "Brinjal",
    "soyabean": "Soybean",
    "soybean": "Soybean",
    "pigeon_pea": "Pigeon_Pea",
    "pigeon pea": "Pigeon_Pea",
}

PEST_NAMES = {
    "aphid": "Aphid",
    "jassid": "Jassid",
    "whitefly": "Whitefly",
    "white_fly": "Whitefly",
    "mealy_bug": "Mealy_Bug",
    "mealy_bugs": "Mealy_Bug",
    "mealybug": "Mealy_Bug",
    "mealybugs": "Mealy_Bug",
    "thrips": "Thrips",
    "red_cotton_bug": "Red_Cotton_Bug",
    "dusky_cotton_bug": "Dusky_Cotton_Bug",
    "leaf_roller": "Leaf_Roller",
    "pink_bollworm": "Pink_Bollworm",
    "fall_army_worm": "Fall_Army_Worm",
    "leaf_miner": "Leaf_Miner",
    "pea_leaf_miner": "Leaf_Miner",
    "pod_borer": "Pod_Borer",
    "hopper": "Hopper",
    "fruit_borer": "Fruit_Borer",
    "shoot_and_fruit_borer": "Fruit_Borer",
    "hairy_caterpiller": "Hairy_Caterpillar",
    "hairy_caterpillar": "Hairy_Caterpillar",
    "healthy": "Healthy",
    "healthy_leaf": "Healthy",
    "healthy_twigs": "Healthy",
    "healthy_fruit": "Healthy",
}

SPECIES_ID = {
    "Aphid": "aphid_sp",
    "Jassid": "jassid_sp",
    "Whitefly": "whitefly_sp",
    "Mealy_Bug": "mealy_bug_sp",
    "Thrips": "thrips_sp",
    "Red_Cotton_Bug": "red_cotton_bug_sp",
    "Dusky_Cotton_Bug": "dusky_cotton_bug_sp",
    "Leaf_Roller": "leaf_roller_sp",
    "Pink_Bollworm": "pink_bollworm_sp",
    "Fall_Army_Worm": "fall_army_worm_sp",
    "Leaf_Miner": "leaf_miner_sp",
    "Pod_Borer": "pod_borer_sp",
    "Hopper": "hopper_sp",
    "Fruit_Borer": "fruit_borer_sp",
    "Hairy_Caterpillar": "hairy_caterpillar_sp",
    "Healthy": None,
}

EXISTING_PEST_BODY_NAMES = {"pest", "pest_body"}
EXISTING_SYMPTOM_NAMES = {"symptoms", "symptom"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── Helper functions ─────────────────────────────────────────────────────────

def canonical_crop(name: str) -> Optional[str]:
    """Get canonical crop name."""
    return CROP_NAMES.get(name.lower().replace(" ", "_"))


def canonical_pest(name: str) -> Optional[str]:
    """Get canonical pest name."""
    return PEST_NAMES.get(name.lower().replace(" ", "_"))


def is_image(path: Path) -> bool:
    """Check if file is an image."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_images(folder: Path) -> List[Path]:
    """Recursively collect all image files."""
    return [p for p in folder.rglob("*") if p.is_file() and is_image(p)]


def log(message: str, dry_run: bool = False, level: str = "INFO") -> None:
    """Print log message."""
    prefix = "[DRY RUN] " if dry_run else ""
    level_str = f"[{level}]" if level != "INFO" else ""
    print(f"{prefix}{level_str} {message}")


def copy_image(src: Path, dst: Path, dry_run: bool = False) -> bool:
    """Copy image, handling duplicates."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle duplicates
    if dst.exists():
        stem, suffix = dst.stem, dst.suffix
        i = 1
        while dst.exists():
            dst = dst.parent / f"{stem}_{i}{suffix}"
            i += 1
    
    if not dry_run:
        shutil.copy2(src, dst)
    
    return True


# ── Smart type detection ──────────────────────────────────────────────────────

def detect_predominant_type(
    pest_dir: Path,
    sample_size: int = 20,
) -> Tuple[str, float]:
    """
    Detect predominant image type (pest_body vs symptom) using heuristics.
    
    Heuristics:
    - Green/brown ratio: symptom = more green, pest_body = more brown/varied
    - Contrast: pest images often have higher contrast (sharp edges)
    - Saturation: symptom images more saturated in green channel
    - Edge density: pest images have more distinct edges
    
    Args:
        pest_dir: Directory with pest images
        sample_size: Number of random images to sample
    
    Returns:
        (predominant_type, confidence)
        predominant_type: "pest_body" or "symptom"
        confidence: 0.0-1.0 (how confident the detection is)
    """
    images = collect_images(pest_dir)
    
    if not images:
        return "symptom", 0.5  # Default to symptom if no images
    
    # Sample images
    sample = random.sample(images, min(sample_size, len(images)))
    
    pest_body_score = 0.0
    symptom_score = 0.0
    
    for img_path in sample:
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            
            # Extract channels
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            # Heuristic 1: Green/Brown ratio
            # Symptoms: high green (leaves are green)
            # Pest bodies: brown/tan colors, less green
            green_mean = np.mean(g)
            brown_mean = np.mean(r) + np.mean(b)  # Approximate brown
            green_ratio = green_mean / (brown_mean + 1e-6)
            
            if green_ratio > 1.2:
                symptom_score += 0.4
            else:
                pest_body_score += 0.4
            
            # Heuristic 2: Saturation in green channel
            # Symptoms have high green saturation (healthy leaves)
            green_std = np.std(g)
            if green_std > 30:
                symptom_score += 0.3
            else:
                pest_body_score += 0.3
            
            # Heuristic 3: Contrast (Laplacian edge detection)
            # Pest bodies: insects/mites have distinct edges
            # Symptoms: more diffuse damage
            gray = np.mean(img_array, axis=2)
            edges = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
            edge_density = np.mean(edges)
            
            if edge_density > 15:
                pest_body_score += 0.3
            else:
                symptom_score += 0.3
            
        except Exception:
            continue
    
    # Normalize
    total = pest_body_score + symptom_score
    if total > 0:
        pest_body_score /= total
        symptom_score /= total
    
    # Determine predominant type
    if pest_body_score > symptom_score:
        predominant = "pest_body"
        confidence = pest_body_score
    else:
        predominant = "symptom"
        confidence = symptom_score
    
    return predominant, confidence


# ── Dataset analysis ────────────────────────────────────────────────────────

def analyze_raw_dataset(
    raw_dir: Path,
    detect_types: bool = True,
) -> Dict:
    """
    Analyze raw dataset structure and detect predominant types.
    
    Args:
        raw_dir: Path to raw dataset
        detect_types: If True, auto-detect predominant type for merged classes
    
    Returns:
        Dict with analysis results
    """
    analysis = {
        "crops": defaultdict(lambda: defaultdict(int)),
        "already_split": [],
        "merged": [],  # Now includes detected types
        "unknown_crops": [],
        "unknown_pests": [],
        "total_images": 0,
        "detected_types": {},  # (crop, pest) -> (type, confidence)
    }

    for crop_dir in sorted(raw_dir.iterdir()):
        if not crop_dir.is_dir():
            continue

        crop_canon = canonical_crop(crop_dir.name)
        if crop_canon is None:
            analysis["unknown_crops"].append(crop_dir.name)
            continue

        for pest_dir in sorted(crop_dir.iterdir()):
            if not pest_dir.is_dir():
                continue

            pest_lower = pest_dir.name.lower().replace(" ", "_")
            
            # Check if healthy variant
            if pest_lower in {"healthy", "healthy_leaf", "healthy_twigs", "healthy_fruit"}:
                images = collect_images(pest_dir)
                analysis["crops"][crop_canon]["Healthy"] += len(images)
                analysis["total_images"] += len(images)
                continue

            pest_canon = canonical_pest(pest_lower)
            if pest_canon is None:
                analysis["unknown_pests"].append(f"{crop_canon}/{pest_dir.name}")
                continue

            # Check for existing splits
            subdirs = {d.name.lower(): d for d in pest_dir.iterdir() if d.is_dir()}
            has_split = any(
                d in EXISTING_PEST_BODY_NAMES or d in EXISTING_SYMPTOM_NAMES
                for d in subdirs.keys()
            )

            if has_split:
                # Count images in each subfolder
                for subdir_name, subdir_path in subdirs.items():
                    images = collect_images(subdir_path)
                    if subdir_name in EXISTING_PEST_BODY_NAMES:
                        analysis["crops"][crop_canon][f"{pest_canon}_pest_body"] += len(images)
                        analysis["already_split"].append((crop_canon, pest_canon))
                    elif subdir_name in EXISTING_SYMPTOM_NAMES:
                        analysis["crops"][crop_canon][f"{pest_canon}_symptom"] += len(images)
                    analysis["total_images"] += len(images)
            else:
                # Merged class - detect predominant type
                images = collect_images(pest_dir)
                analysis["crops"][crop_canon][pest_canon] += len(images)
                
                detected_type = "symptom"  # Safe default
                confidence = 0.5
                
                if detect_types and len(images) > 0:
                    detected_type, confidence = detect_predominant_type(pest_dir)
                
                analysis["merged"].append({
                    "crop": crop_canon,
                    "pest": pest_canon,
                    "count": len(images),
                    "detected_type": detected_type,
                    "confidence": confidence,
                })
                analysis["detected_types"][(crop_canon, pest_canon)] = (detected_type, confidence)
                analysis["total_images"] += len(images)

    return analysis


def print_analysis(analysis: Dict) -> None:
    """Print analysis results with detected types."""
    print("\n" + "=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal images: {analysis['total_images']}")
    print(f"Total crops: {len(analysis['crops'])}")
    print(f"Already split classes: {len(set(analysis['already_split']))}")
    print(f"Merged classes (auto-detected): {len(analysis['merged'])}")
    
    if analysis["unknown_crops"]:
        print(f"\n⚠ Unknown crops (will skip): {analysis['unknown_crops']}")
    
    if analysis["unknown_pests"]:
        print(f"\n⚠ Unknown pests (will skip):")
        for p in analysis["unknown_pests"][:5]:
            print(f"  - {p}")

    # Show detected types for merged classes
    if analysis["merged"]:
        print("\n" + "-" * 80)
        print("AUTO-DETECTED PREDOMINANT TYPES FOR MERGED CLASSES:")
        print("-" * 80)
        print(f"{'Crop':<15} {'Pest':<25} {'Count':>6} {'Type':<12} {'Confidence':>8}")
        print("-" * 80)
        
        for item in sorted(analysis["merged"], key=lambda x: (x["crop"], x["pest"])):
            crop = item["crop"]
            pest = item["pest"]
            count = item["count"]
            dtype = item["detected_type"]
            conf = item["confidence"]
            
            confidence_str = f"{conf:.1%}"
            print(f"{crop:<15} {pest:<25} {count:>6} {dtype:<12} {confidence_str:>8}")
    
    print("\n" + "=" * 80)


def prompt_for_overrides(analysis: Dict) -> Dict[Tuple[str, str], Optional[str]]:
    """
    Prompt user to confirm or override auto-detected types.
    
    Returns:
        Dict mapping (crop, pest) -> "pest_body" | "symptom" | None
    """
    print("\n" + "=" * 80)
    print("CONFIRM OR OVERRIDE AUTO-DETECTED TYPES")
    print("=" * 80)
    print("\nFor each class, press ENTER to accept detected type, or enter override:")
    print("  pest_body  - use pest_body as default")
    print("  symptom    - use symptom as default")
    print("  skip       - exclude this class\n")

    selections = {}

    for item in sorted(analysis["merged"], key=lambda x: (x["crop"], x["pest"])):
        crop = item["crop"]
        pest = item["pest"]
        detected = item["detected_type"]
        confidence = item["confidence"]
        count = item["count"]

        print(f"{crop}/{pest} ({count} images)")
        print(f"  Auto-detected: {detected} ({confidence:.0%} confidence)")
        
        while True:
            choice = input(f"  Confirm? [{detected}/pest_body/symptom/skip]: ").strip().lower()
            
            if choice == "" or choice == detected:
                selections[(crop, pest)] = detected
                print(f"  ✓ Using: {detected}\n")
                break
            elif choice in ("pest_body", "symptom"):
                selections[(crop, pest)] = choice
                print(f"  ✓ Override: {choice}\n")
                break
            elif choice == "skip":
                selections[(crop, pest)] = None
                print(f"  ✓ Skipping\n")
                break
            else:
                print("  Invalid choice. Try again.")

    return selections


# ── Main processing ────────────────────────────────────────────────────────

def process_dataset(
    raw_dir: Path,
    processed_dir: Path,
    selections: Dict[Tuple[str, str], Optional[str]],
    dry_run: bool = False,
) -> Tuple[List[Dict], List[str]]:
    """
    Process raw dataset into cleaned structure.
    
    Args:
        raw_dir: Path to raw dataset
        processed_dir: Path to output
        selections: Type selections for merged classes
        dry_run: If True, don't actually copy files
    
    Returns:
        (records for labels.csv, list of warnings)
    """
    records = []
    warnings = []

    if not dry_run:
        processed_dir.mkdir(parents=True, exist_ok=True)

    log(f"Processing dataset...", dry_run)

    for crop_dir in sorted(raw_dir.iterdir()):
        if not crop_dir.is_dir():
            continue

        crop_canon = canonical_crop(crop_dir.name)
        if crop_canon is None:
            continue

        for pest_dir in sorted(crop_dir.iterdir()):
            if not pest_dir.is_dir():
                continue

            pest_lower = pest_dir.name.lower().replace(" ", "_")

            # ── Healthy ──────────────────────────────────────────────────
            if pest_lower in {"healthy", "healthy_leaf", "healthy_twigs", "healthy_fruit"}:
                images = collect_images(pest_dir)
                
                for img in images:
                    dst_path = processed_dir / crop_canon / "Healthy" / img.name
                    copy_image(img, dst_path, dry_run)
                    
                    records.append({
                        "filepath": str(dst_path.relative_to(processed_dir)),
                        "crop": crop_canon,
                        "pest": "Healthy",
                        "image_type": "healthy",
                        "species_id": None,
                        "split": "",
                    })
                
                log(f"  {crop_canon}/Healthy: {len(images)} images", dry_run)
                continue

            # ── Pest ──────────────────────────────────────────────────
            pest_canon = canonical_pest(pest_lower)
            if pest_canon is None:
                continue

            species = SPECIES_ID.get(pest_canon, f"{pest_lower}_sp")

            # Check for existing splits
            subdirs = {d.name.lower(): d for d in pest_dir.iterdir() if d.is_dir()}
            has_split = any(
                d in EXISTING_PEST_BODY_NAMES or d in EXISTING_SYMPTOM_NAMES
                for d in subdirs.keys()
            )

            if has_split:
                # ── Already split ─────────────────────────────────────
                for subdir_name, subdir_path in subdirs.items():
                    images = collect_images(subdir_path)
                    
                    if subdir_name in EXISTING_PEST_BODY_NAMES:
                        img_type = "pest_body"
                    elif subdir_name in EXISTING_SYMPTOM_NAMES:
                        img_type = "symptom"
                    else:
                        continue
                    
                    for img in images:
                        dst_path = processed_dir / crop_canon / pest_canon / img_type / img.name
                        copy_image(img, dst_path, dry_run)
                        
                        records.append({
                            "filepath": str(dst_path.relative_to(processed_dir)),
                            "crop": crop_canon,
                            "pest": pest_canon,
                            "image_type": img_type,
                            "species_id": species,
                            "split": "",
                        })
                    
                    log(f"  {crop_canon}/{pest_canon}/{img_type}: {len(images)} images", dry_run)

            else:
                # ── Merged, use selection ─────────────────────────────
                selected_type = selections.get((crop_canon, pest_canon))
                
                if selected_type is None:
                    log(f"  {crop_canon}/{pest_canon}: SKIPPED (user choice)", dry_run)
                    continue
                
                images = collect_images(pest_dir)
                
                for img in images:
                    dst_path = processed_dir / crop_canon / pest_canon / selected_type / img.name
                    copy_image(img, dst_path, dry_run)
                    
                    records.append({
                        "filepath": str(dst_path.relative_to(processed_dir)),
                        "crop": crop_canon,
                        "pest": pest_canon,
                        "image_type": selected_type,
                        "species_id": species,
                        "split": "",
                    })
                
                log(f"  {crop_canon}/{pest_canon}: {len(images)} images → {selected_type}/", dry_run)

    return records, warnings


def assign_splits(
    records: List[Dict],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> List[Dict]:
    """Assign train/val/test splits stratified by (crop, pest, image_type)."""
    random.seed(seed)
    
    buckets = defaultdict(list)
    for r in records:
        key = (r["crop"], r["pest"], r["image_type"])
        buckets[key].append(r)
    
    for key, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        
        for i, item in enumerate(items):
            if i < n_train:
                item["split"] = "train"
            elif i < n_train + n_val:
                item["split"] = "val"
            else:
                item["split"] = "test"
    
    return records


def write_labels_csv(records: List[Dict], processed_dir: Path) -> Path:
    """Write labels.csv file."""
    output_path = processed_dir / "labels.csv"
    
    fieldnames = ["filepath", "crop", "pest", "image_type", "species_id", "split"]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    return output_path


def write_summary(records: List[Dict], processed_dir: Path) -> Path:
    """Write dataset summary report."""
    output_path = processed_dir / "dataset_summary.txt"
    
    lines = []
    lines.append("=" * 70)
    lines.append("DATASET PREPARATION SUMMARY")
    lines.append("=" * 70)
    
    lines.append(f"\nTotal images: {len(records)}")
    
    crop_counts = Counter(r["crop"] for r in records)
    lines.append(f"Total crops: {len(crop_counts)}")
    
    pest_counts = Counter(r["pest"] for r in records)
    lines.append(f"Total pest classes: {len([p for p in pest_counts if p != 'Healthy'])}")
    
    type_counts = Counter(r["image_type"] for r in records)
    lines.append(f"\nImage type distribution:")
    for img_type, count in sorted(type_counts.items()):
        lines.append(f"  {img_type:<15} {count:>6} ({100*count/len(records):>5.1f}%)")
    
    split_counts = Counter(r["split"] for r in records)
    lines.append(f"\nTrain/Val/Test split:")
    for split in ["train", "val", "test"]:
        count = split_counts.get(split, 0)
        lines.append(f"  {split:<10} {count:>6} ({100*count/len(records):>5.1f}%)")
    
    lines.append(f"\nPer-crop breakdown:")
    for crop in sorted(crop_counts.keys()):
        crop_records = [r for r in records if r["crop"] == crop]
        pest_types = set((r["pest"], r["image_type"]) for r in crop_records)
        lines.append(f"\n{crop}:")
        lines.append(f"  Total images: {len(crop_records)}")
        lines.append(f"  Pest/symptom types: {len(pest_types)}")
        
        for pest, img_type in sorted(pest_types):
            count = sum(1 for r in crop_records if r["pest"] == pest and r["image_type"] == img_type)
            lines.append(f"    {pest:<25} {img_type:<12} {count:>5}")
    
    lines.append("\n" + "=" * 70)
    lines.append("NEXT STEPS")
    lines.append("=" * 70)
    lines.append("\n1. Verify dataset structure:")
    lines.append("   find data/processed -type f | head -20")
    lines.append("\n2. Validate dataset:")
    lines.append("   python scripts/validate_dataset.py --dataset data/processed")
    lines.append("\n3. Start Phase 1 training:")
    lines.append("   torchrun --nproc_per_node=4 scripts/train_phase1.py --config-dir configs")
    
    content = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(content)
    
    return output_path


# ── Main entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare raw pest dataset for training with smart type detection"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default="data/raw",
        help="Path to raw dataset",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default="data/processed",
        help="Path to output processed dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying files",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt to confirm/override auto-detected types (interactive mode)",
    )
    parser.add_argument(
        "--default",
        choices=["symptom", "pest_body"],
        default=None,
        help="Force all merged classes to this type (overrides auto-detection)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite if output directory exists",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    processed_dir = Path(args.processed)

    # Validate inputs
    if not raw_dir.exists():
        print(f"ERROR: Raw dataset directory not found: {raw_dir}")
        sys.exit(1)

    if processed_dir.exists() and not args.force and not args.dry_run:
        print(f"ERROR: Output directory already exists: {processed_dir}")
        print(f"Use --force to overwrite or --dry-run to preview")
        sys.exit(1)

    if not args.dry_run:
        abs_sum = args.train_split + args.val_split + args.test_split
        if abs(abs_sum - 1.0) > 0.01:
            print(f"ERROR: Split fractions must sum to 1.0 (got {abs_sum})")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("PEST DATASET PREPARATION WITH SMART TYPE DETECTION")
    print("=" * 80)
    print(f"\nRaw dataset:       {raw_dir}")
    print(f"Processed output:  {processed_dir}")
    print(f"Dry run:           {args.dry_run}")
    
    if args.default:
        print(f"Default type:      {args.default} (force all merged classes)")
    elif args.interactive:
        print(f"Mode:              Interactive (confirm/override auto-detected types)")
    else:
        print(f"Mode:              Auto-detect predominant types (non-interactive)")

    # Analyze dataset with auto-detection
    print("\nAnalyzing raw dataset and detecting predominant types...")
    analysis = analyze_raw_dataset(raw_dir, detect_types=True)
    print_analysis(analysis)

    # Get selections
    if args.default:
        # Force all to default type
        selections = {(item["crop"], item["pest"]): args.default for item in analysis["merged"]}
        print(f"\nForcing all merged classes to: {args.default}")
    elif args.interactive:
        # Prompt user to confirm/override
        selections = prompt_for_overrides(analysis)
    else:
        # Use auto-detected types (non-interactive)
        selections = {(item["crop"], item["pest"]): item["detected_type"] for item in analysis["merged"]}
        print(f"\nUsing auto-detected predominant types (non-interactive mode)")

    # Process dataset
    print("\n" + "=" * 80)
    print("PROCESSING")
    print("=" * 80 + "\n")
    
    records, warnings = process_dataset(raw_dir, processed_dir, selections, args.dry_run)

    # Assign splits
    records = assign_splits(
        records,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    # Write outputs
    if not args.dry_run:
        labels_csv = write_labels_csv(records, processed_dir)
        summary = write_summary(records, processed_dir)
        
        print("\n" + "=" * 80)
        print("COMPLETE")
        print("=" * 80)
        print(f"\n✓ Dataset processed successfully!")
        print(f"  Images copied: {len(records)}")
        print(f"  Output directory: {processed_dir}")
        print(f"  Labels CSV: {labels_csv}")
        print(f"  Summary: {summary}\n")
    else:
        print("\n" + "=" * 80)
        print("[DRY RUN]")
        print("=" * 80)
        print(f"\nWould process {len(records)} images")
        print(f"Would create labels.csv in {processed_dir}")
        print(f"\nRun without --dry-run to execute")


if __name__ == "__main__":
    main()