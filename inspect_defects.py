import json
from pathlib import Path

po = json.loads(Path("nisarg_shared/parts_override.json").read_text())
ms = next((json.loads(Path(p).read_text())
           for p in ("master_suite.json", "nisarg_shared/master_suite.json")
           if Path(p).exists()), None)
prompts = ms["prompts"] if ms else None

DEFECTS = ["052", "117", "118", "194", "293"]
NEIGH = 2
for d in DEFECTS:
    di = int(d)
    print("=" * 72)
    for j in range(di - NEIGH, di + NEIGH + 1):
        key = f"{j:03d}"
        if key not in po:
            continue
        pr = prompts[j] if prompts and j < len(prompts) else "(prompt unavailable)"
        print(f"idx {key}{'   <-- DEFECT' if key == d else ''}")
        print(f"    prompt: {pr}")
        print(f"    parts : {po[key]}")
    print()
