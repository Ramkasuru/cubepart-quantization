"""
One-time fix: halves the chamfer distances in per-mesh JSON files written when
chamfer_distance() returned the sum of the two directions instead of the average.
Files are stamped "cd_convention": "average" and skipped on a rerun.
"""

import json
from pathlib import Path

PER_MESH_DIR = Path("results/full_sweep_4config/per_mesh")


def migrate_file(path: Path) -> bool:
    data = json.loads(path.read_text())
    if data.get("cd_convention") == "average":
        return False  # already migrated
    if "chamfer_distance_per_part" not in data:
        return False  # error entry, nothing to migrate

    data["chamfer_distance_per_part"] = [
        round(c / 2.0, 5) if c is not None else None
        for c in data["chamfer_distance_per_part"]
    ]
    if data.get("chamfer_distance_mean") is not None:
        data["chamfer_distance_mean"] = round(data["chamfer_distance_mean"] / 2.0, 5)
    data["cd_convention"] = "average"

    path.write_text(json.dumps(data, indent=2))
    return True


def main():
    if not PER_MESH_DIR.exists():
        print(f"{PER_MESH_DIR} does not exist yet -- nothing to migrate.")
        return

    migrated, skipped = 0, 0
    for path in sorted(PER_MESH_DIR.glob("*.json")):
        if migrate_file(path):
            migrated += 1
        else:
            skipped += 1

    print(f"Migrated {migrated} files, skipped {skipped} (already migrated or error entries).")


if __name__ == "__main__":
    main()
