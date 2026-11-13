"""
Flags parts_override.json entries for manual review. Nothing is fixed automatically.

Checks:
  1. a part name that marks a specific animal (shell, mane, horns, tusks) when the prompt
     does not mention that animal (high severity)
  2. unusual part count, duplicate names, left/right names, generic names (main body, part)
  3. no word overlap with the prompt (weak, skipped for parts caught by 1)
  4. same parts list on 4 or more prompts

Fetches master_suite.json from Hugging Face if it is not cached, so run it where there is internet.
The full report goes to spot_check_output.txt, high severity flags are also printed.
"""

import argparse
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

PARTS_OVERRIDE = Path("nisarg_shared/parts_override.json")
MASTER_SUITE_CANDIDATES = [
    Path("master_suite.json"),
    Path("nisarg_shared/master_suite.json"),
]
MASTER_SUITE_URL = "https://huggingface.co/TrNi/efficient-cube3d/raw/main/master_suite.json"

SPATIAL_WORDS = {"left", "right"}
SOFT_SPATIAL_WORDS = {"front", "back"}
GENERIC_NAMES = {"main body", "part", "piece", "object", "thing", "top", "bottom", "body"}
STOPWORDS = {"a", "an", "the", "with", "of", "and", "for", "on", "in", "to"}

# animal part -> words expected in the prompt
ANATOMY_MARKERS = {
    "shell": {"turtle", "tortoise", "snail", "crab", "shellfish", "armadillo", "clam", "beetle"},
    "mane": {"horse", "lion", "zebra", "pony"},
    "horns": {"goat", "bull", "cow", "ram", "deer", "rhino", "ox", "antelope", "buffalo", "cattle", "bison"},
    "antlers": {"deer", "moose", "elk", "reindeer", "stag"},
    "tusks": {"elephant", "walrus", "boar", "warthog", "mammoth"},
    "trunk": {"elephant"},
    "hump": {"camel", "dromedary"},
    "pouch": {"kangaroo", "koala", "wallaby"},
    "beak": {"bird", "duck", "rooster", "chicken", "parrot", "eagle", "owl", "penguin", "goose", "swan", "hen", "peacock", "toucan"},
    "feathers": {"bird", "duck", "rooster", "chicken", "parrot", "eagle", "owl", "penguin", "goose", "swan", "hen", "peacock"},
    "fins": {"fish", "shark", "dolphin", "whale"},
    "gills": {"fish", "shark"},
    "scales": {"fish", "snake", "lizard", "reptile", "dragon", "iguana"},
    "udder": {"cow", "goat", "cattle"},
    "wool": {"sheep", "lamb", "alpaca", "llama"},
    "whiskers": {"cat", "mouse", "rat", "seal", "walrus", "kitten"},
    "snout": {"pig", "dog", "alligator", "crocodile", "anteater", "boar", "hog"},
    "claws": {"cat", "crab", "lobster", "bear", "lion", "tiger", "eagle", "hawk"},
}


def load_master_suite():
    for path in MASTER_SUITE_CANDIDATES:
        if path.exists():
            return json.loads(path.read_text())
    try:
        with urllib.request.urlopen(MASTER_SUITE_URL, timeout=10) as resp:
            data = json.loads(resp.read())
        Path("master_suite.json").write_text(json.dumps(data))
        print(">> Downloaded and cached master_suite.json locally")
        return data
    except Exception as e:
        print(f">> Could not load master_suite.json ({e}) -- skipping prompt "
              f"word-overlap and anatomy checks, running the rest\n")
        return None


def tokenize(text: str):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="spot_check_output.txt")
    args = ap.parse_args()

    parts_override = json.loads(PARTS_OVERRIDE.read_text())
    master = load_master_suite()

    flags = defaultdict(list)
    high_severity = []  # (idx, prompt, part, expected_animals)
    signature_count = Counter()

    for idx, parts in sorted(parts_override.items()):
        n = len(parts)
        if n < 2 or n > 10:
            flags[idx].append(f"unusual part count: {n} (outside 2-10 entirely)")
        elif n > 5:
            flags[idx].append(f"part count {n} is in the rarer 6-10 bucket (~20% of assets) -- confirm intentional")

        lowered = [p.lower().strip() for p in parts]
        if len(set(lowered)) != len(lowered):
            flags[idx].append("duplicate part names within this entry")

        prompt = master["prompts"][int(idx)] if master is not None else None
        prompt_words = tokenize(prompt) if prompt else set()

        for p in lowered:
            words = set(p.split())
            anatomy_flagged = False

            if words & SPATIAL_WORDS:
                flags[idx].append(f"left/right spatial identifier in {p!r} -- Section 7's named VLM failure mode")
            elif words & SOFT_SPATIAL_WORDS:
                flags[idx].append(f"front/back in {p!r} -- related spatial-identifier concern, softer flag")
            if p in GENERIC_NAMES:
                flags[idx].append(f"generic/low-information part name: {p!r}")

            if prompt is not None:
                for marker, expected_animals in ANATOMY_MARKERS.items():
                    if marker in words:
                        if not (expected_animals & prompt_words):
                            msg = f"ANATOMICALLY IMPLAUSIBLE: {p!r} (marker {marker!r}) vs prompt {prompt!r} -- expected one of {sorted(expected_animals)}"
                            flags[idx].append(msg)
                            high_severity.append((idx, prompt, p, sorted(expected_animals)))
                            anatomy_flagged = True

                if not anatomy_flagged:
                    part_words = tokenize(p)
                    if part_words and not (part_words & prompt_words):
                        flags[idx].append(
                            f"no word overlap with prompt: {p!r} vs prompt {prompt!r} (soft signal, often fine)"
                        )

        signature_count[tuple(lowered)] += 1

    lines = []
    lines.append(f"Checked {len(parts_override)} entries.\n")

    lines.append(f"=== HIGH SEVERITY: {len(high_severity)} anatomically implausible parts ===\n")
    for idx, prompt, part, expected in high_severity:
        lines.append(f"idx={idx}  prompt={prompt!r}  part={part!r}  expected one of {expected}")
    lines.append("")

    reused = {sig: count for sig, count in signature_count.items() if count >= 4}
    if reused:
        lines.append(f"=== Exact parts list reused across 4+ different indices ({len(reused)} such lists) ===")
        for sig, count in sorted(reused.items(), key=lambda x: -x[1]):
            lines.append(f"  {count}x  {list(sig)}")
        lines.append("")

    lines.append(f"=== Full per-entry detail: {len(flags)} of {len(parts_override)} entries have at least one flag ===\n")
    for idx, msgs in sorted(flags.items()):
        lines.append(f"idx={idx}  parts={parts_override[idx]}")
        for m in msgs:
            lines.append(f"    - {m}")
        lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote full report to {args.out} ({len(lines)} lines)\n")

    print(f"=== HIGH SEVERITY: {len(high_severity)} anatomically implausible parts (also in {args.out}) ===")
    for idx, prompt, part, expected in high_severity:
        print(f"idx={idx}  prompt={prompt!r}  part={part!r}  expected one of {expected}")


if __name__ == "__main__":
    main()
