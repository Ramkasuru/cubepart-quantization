"""
Aggregates CD(A,B) and CD(C,D) from results/full_sweep_4config/per_mesh/ into min/max/mean/median
per prompt, per category and for the whole dataset. Categories come from fixed index ranges.
"""

import argparse
import json
import statistics as stats
from pathlib import Path

PER_MESH_DIR = Path("results/full_sweep_4config/per_mesh")
OUT_PATH = Path("results/full_sweep_4config/cd_stats.json")

# (start_idx, end_idx, category). abstract_mathematical, geometric_primitive and
# symmetry_topology are excluded.
CATEGORY_RANGES = [
    (20, 39, "animal_domestic"),
    (40, 59, "animal_wild"),
    (60, 79, "architecture"),
    (80, 99, "electronics"),
    (100, 119, "fine_detail"),
    (120, 139, "furniture"),
    (160, 179, "musical_instrument"),
    (180, 199, "nature_plant"),
    (220, 239, "tool_hardware"),
    (240, 259, "vehicle_air_water"),
    (260, 279, "vehicle_land"),
    (280, 309, "original_visuals"),
]


def category_for_idx(idx: int) -> str:
    for lo, hi, name in CATEGORY_RANGES:
        if lo <= idx <= hi:
            return name
    return "unknown"


def summarize(values):
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "mean": stats.fmean(values),
        "median": stats.median(values),
        "n": len(values),
    }


def collect(suffix: str):
    """suffix is '_ab' or '_cd'. Returns rows of (idx, category, cd)."""
    rows = []
    for path in sorted(PER_MESH_DIR.glob(f"*{suffix}.json")):
        data = json.loads(path.read_text())
        if "error" in data or "chamfer_distance_per_part" not in data:
            continue
        idx = int(data["idx"])
        category = category_for_idx(idx)
        for cd in data["chamfer_distance_per_part"]:
            if cd is not None:
                rows.append((data["idx"], category, cd))
    return rows


def build_report(rows):
    by_prompt, by_category = {}, {}
    for idx, category, cd in rows:
        by_prompt.setdefault(idx, []).append(cd)
        by_category.setdefault(category, []).append(cd)
    return {
        "per_prompt": {k: summarize(v) for k, v in by_prompt.items()},
        "per_category": {k: summarize(v) for k, v in by_category.items()},
        "dataset": summarize([cd for _, _, cd in rows]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    ab_rows = collect("_ab")
    cd_rows = collect("_cd")

    report = {
        "CD(A,B)": build_report(ab_rows),
        "CD(C,D)": build_report(cd_rows),
    }

    print(f"CD(A,B): n_prompts={len(report['CD(A,B)']['per_prompt'])} dataset={report['CD(A,B)']['dataset']}")
    print(f"CD(C,D): n_prompts={len(report['CD(C,D)']['per_prompt'])} dataset={report['CD(C,D)']['dataset']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
