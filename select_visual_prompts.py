"""
Pick 3 representative prompts per category for Nisarg's paper-figure asset
package (his ask, 2026-09-18): "for 3 prompts per category... prompt ids,
prompt text, parts list".

Selection rule: per category, the 3 meshes whose CD(A,B) mean is closest to
that category's median -- i.e. the most TYPICAL cases, not outliers. The
point of the figure is to show the decomposition looking clean/near-lossless;
an outlier-driven pick would undermine that unless Nisarg specifically wants
a limitations figure (ask him if so).

Run from ~/cube/cubepart (needs master_suite.json, nisarg_shared/parts_override.json,
results/full_sweep_4config/per_mesh/*_ab.json from the clean sweep, job 1159996).

  python select_visual_prompts.py --out visual_prompts_36.json
"""
import argparse
import json
import statistics as stats
from pathlib import Path

PER_MESH = Path("results/full_sweep_4config/per_mesh")
PARTS_OVERRIDE = Path("nisarg_shared/parts_override.json")
MASTER_SUITE_CANDIDATES = [Path("master_suite.json"), Path("nisarg_shared/master_suite.json")]

N_PER_CATEGORY = 3


def load_master_suite():
    for p in MASTER_SUITE_CANDIDATES:
        if p.exists():
            return json.loads(p.read_text())
    raise SystemExit("master_suite.json not found -- run spot_check_parts_override.py first to fetch it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="visual_prompts_36.json")
    ap.add_argument("--n", type=int, default=N_PER_CATEGORY)
    ap.add_argument("--pin", type=str, default="",
                    help="comma separated idx that must be included, e.g. 130,060,020 (already delivered)")
    args = ap.parse_args()
    pinned = {p.strip() for p in args.pin.split(",") if p.strip()}

    po = json.loads(PARTS_OVERRIDE.read_text())
    ms = load_master_suite()
    prompts, cats = ms["prompts"], ms["categories"]

    by_cat = {}
    for k in po:
        i = int(k)
        ab_path = PER_MESH / f"{k}_ab.json"
        if not ab_path.exists():
            continue
        d = json.loads(ab_path.read_text())
        cd = d.get("chamfer_distance_mean")
        if cd is None:
            continue
        c = cats[i]
        by_cat.setdefault(c, []).append((k, i, cd))

    selected = {}
    for c, rows in sorted(by_cat.items()):
        cds = [r[2] for r in rows]
        med = stats.median(cds)
        rows_sorted = sorted(rows, key=lambda r: abs(r[2] - med))
        pin_rows = [r for r in rows if r[0] in pinned]
        rest = [r for r in rows_sorted if r[0] not in pinned]
        picked = (pin_rows + rest)[: args.n]
        selected[c] = [
            {
                "idx": k,
                "prompt": prompts[i],
                "parts": po[k],
                "cd_ab_mean": cd,
                "category_median_cd": med,
            }
            for k, i, cd in sorted(picked, key=lambda r: r[0])
        ]

    total = sum(len(v) for v in selected.values())
    print(f"Selected {total} prompts across {len(selected)} categories (target {args.n} each)")
    for c, rows in selected.items():
        print(f"  {c}: " + ", ".join(r["idx"] for r in rows))

    Path(args.out).write_text(json.dumps(selected, indent=2))
    print(f"\nWrote {args.out}")
    print("Override specific picks by hand-editing this file before running export_visual_assets.py")


if __name__ == "__main__":
    main()
