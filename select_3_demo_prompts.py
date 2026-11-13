"""
Picks 3 prompts (furniture, architecture, animal_domestic) for the 3 prompts x 4 models example.
Parts are read from parts_override.json.

    python select_3_demo_prompts.py --out visual_prompts_3.json
"""
import argparse
import json
from pathlib import Path

# bar stool, medieval castle, sitting cat
PICKS = ["130", "060", "020"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="visual_prompts_3.json")
    args = ap.parse_args()

    po = json.loads(Path("nisarg_shared/parts_override.json").read_text())
    ms = json.loads(Path("master_suite.json").read_text())
    prompts, cats = ms["prompts"], ms["categories"]

    out = {}
    for k in PICKS:
        i = int(k)
        out.setdefault(cats[i], []).append({"idx": k, "prompt": prompts[i], "parts": po[k]})

    Path(args.out).write_text(json.dumps(out, indent=2))
    for c, rows in out.items():
        for r in rows:
            print(f"{r['idx']}  [{c}]  {r['prompt']!r}")
            print(f"      parts={r['parts']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
