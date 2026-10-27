import json, re
from pathlib import Path
from collections import defaultdict

STOP = {"and","the","with","for","its","this","that","has","have","are"}
def tok(s): return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2 and w not in STOP}

po = json.loads(Path("nisarg_shared/parts_override.json").read_text())
ms = next((json.loads(Path(p).read_text())
           for p in ("master_suite.json", "nisarg_shared/master_suite.json")
           if Path(p).exists()), None)
assert ms, "master_suite.json missing -- run spot_check_parts_override.py first to fetch it"
prompts = ms["prompts"]

CATS = [(20,39,"animal_domestic"),(40,59,"animal_wild"),(60,79,"architecture"),
        (80,99,"electronics"),(100,119,"fine_detail"),(120,139,"furniture"),
        (160,179,"musical_instrument"),(180,199,"nature_plant"),
        (220,239,"tool_hardware"),(240,259,"vehicle_air_water"),
        (260,279,"vehicle_land"),(280,309,"original_visuals")]
def cat(i):
    for lo,hi,n in CATS:
        if lo <= i <= hi: return n
    return "?"

ptok = {int(k): tok(prompts[int(k)]) for k in po}
rows, per_cat = [], defaultdict(lambda: [0, 0])
for k in sorted(po, key=int):
    i = int(k)
    ov = len(tok(" ".join(po[k])) & ptok[i])
    rows.append((cat(i), k, i, ov, po[k]))
    per_cat[cat(i)][0] += 1
    per_cat[cat(i)][1] += (ov == 0)

print("PER-CATEGORY prompt<->parts word overlap  (high zero% => that block is misaligned)")
print(f"{'category':22}{'n':>4}{'zero-ov':>9}{'zero%':>7}")
for _, _, n in CATS:
    t, z = per_cat[n]
    if t: print(f"{n:22}{t:>4}{z:>9}{100*z//t:>6}%")

print("\nZERO-OVERLAP ENTRIES (eyeball each; terse-but-correct prompts also land here):")
for c, k, i, ov, parts in rows:
    if ov: continue
    pt = tok(" ".join(parts))
    cand = [(s, j) for s, j in sorted(((len(pt & ptok[j]), j) for j in ptok if j != i), reverse=True)[:3] if s >= 2]
    hint = ("  -> parts better match prompt " + ", ".join(f"{j}(+{s})" for s, j in cand)) if cand else ""
    print(f"[{c}] {k}  {parts}{hint}")
    print(f"        {prompts[i]}")

print("\nFULL DUMP (for a by-eye pass with Nisarg):")
for c, k, i, ov, parts in rows:
    print(f"[{c}] {k} ov={ov}  {parts}")
    print(f"        {prompts[i]}")
