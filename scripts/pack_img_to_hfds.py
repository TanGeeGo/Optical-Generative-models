import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset, DatasetDict, Features, ClassLabel, Image, Value
from sklearn.model_selection import train_test_split
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}

def list_images(root: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def build_records(
    files: List[Path],
    root: Path,
    use_subdir_as_label: bool
) -> Tuple[dict, list]:
    records = {
        "image": [],
        "path": [],
        "filename": [],
    }
    labels = []

    for p in files:
        records["image"].append(str(p))              
        records["path"].append(str(p))
        records["filename"].append(p.name)
        if use_subdir_as_label:
            # label is assumed to be the immediate parent folder name
            rel = p.parent.relative_to(root)
            label_name = rel.parts[-1] if len(rel.parts) > 0 else "_"
            labels.append(label_name)

    return records, labels

def to_datasetdict(
    records: dict,
    labels: list,
    out_dir: Path,
    splits: Tuple[float, float, float],
    seed: int
) -> DatasetDict:
    has_labels = len(labels) > 0
    if has_labels:
        
        unique = sorted(set(labels))
        name2id = {n: i for i, n in enumerate(unique)}
        y = np.array([name2id[n] for n in labels], dtype=np.int64)
        records["label"] = y.tolist()

        features = Features({
            "image": Image(),
            "path": Value("string"),
            "filename": Value("string"),
            "label": ClassLabel(names=unique),
        })

        
        with open(out_dir / "label2id.json", "w", encoding="utf-8") as f:
            json.dump({k: int(v) for k, v in name2id.items()}, f, ensure_ascii=False, indent=2)
    else:
        features = Features({
            "image": Image(),
            "path": Value("string"),
            "filename": Value("string"),
        })

    ds = Dataset.from_dict(records, features=features)

    # split
    train_ratio, val_ratio, test_ratio = splits
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Split ratios must sum to 1."

    idx_all = np.arange(len(ds))
    if len(idx_all) == 0:
        raise RuntimeError("No images found to build a dataset.")

    # test part
    if test_ratio > 0:
        idx_tmp, idx_test = train_test_split(
            idx_all, test_size=test_ratio, random_state=seed, shuffle=True
        )
    else:
        idx_tmp, idx_test = idx_all, np.array([], dtype=int)

    # validation part
    if val_ratio > 0:
        # validation is a part of the remaining train split
        rel_val = val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio) > 0 else 0.0
        idx_train, idx_val = train_test_split(
            idx_tmp, test_size=rel_val, random_state=seed, shuffle=True
        )
    else:
        idx_train, idx_val = idx_tmp, np.array([], dtype=int)

    dsd = DatasetDict({
        "train": ds.select(idx_train)
    })
    if len(idx_val) > 0:
        dsd["validation"] = ds.select(idx_val)
    if len(idx_test) > 0:
        dsd["test"] = ds.select(idx_test)

    return dsd

def main():
    parser = argparse.ArgumentParser(
        description="Pack a folder of images into a Hugging Face Dataset saved to disk."
    )
    parser.add_argument("--src_dir", type=str, help="Folder containing images")
    parser.add_argument("--out_dir", type=str, help="Output directory for .save_to_disk")
    parser.add_argument("--no-recursive", action="store_true", help="Do NOT scan subfolders")
    parser.add_argument("--use-subdir-as-label", action="store_true",
                        help="Use immediate parent folder name as label")
    parser.add_argument("--train", type=float, default=1.0, help="Train split ratio (default: 1.0)")
    parser.add_argument("--val", type=float, default=0.0, help="Validation split ratio")
    parser.add_argument("--test", type=float, default=0.0, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    src = Path(args.src_dir).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    files = list_images(src, recursive=not args.no_recursive)
    if not files:
        raise SystemExit(f"No images found in: {src}")

    records, labels = build_records(files, src, use_subdir_as_label=args.use_subdir_as_label)

    dsd = to_datasetdict(
        records=records,
        labels=labels if args.use_subdir_as_label else [],
        out_dir=out,
        splits=(args.train, args.val, args.test),
        seed=args.seed
    )

    dsd.save_to_disk(str(out))
    print(f"Saved dataset to: {out}")
    print("You can load it with:\n"
          f"  from datasets import load_from_disk\n"
          f"  dsd = load_from_disk(r'{out}')\n"
          f"  print(dsd)")

if __name__ == "__main__":
    main()