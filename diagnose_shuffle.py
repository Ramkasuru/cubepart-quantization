import json, re
from pathlib import Path
from collections import Counter
from statistics import mean

STOP = {"and","the","with","for","its","this","that","has","have","are","two","four","full"}
def tok(s): return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2 and w not in STOP}

po = json.loads(Path("nisarg_shared/parts_override.json").read_text())
ms = json.loads(Path("master_suite.json").read_text())
prompts, cats = ms["prompts"], ms["categories"]
ptok = [tok(p) for p in prompts]
def score(pt, j): return len(pt & ptok[j])

rows = []
for k in sorted(po, key=int):
    i = int(k)
    pt = tok(" ".join(po[k]))
    best_j = max(range(len(prompts)), key=lambda j: score(pt, j))
    rows.append((i, cats[i], score(pt, i), score(pt, best_j), best_j, po[k]))

own_best = sum(1 for i,c,ss,bs,bj,pp in rows if ss >= bs and ss > 0)
strong_elsewhere = sum(1 for i,c,ss,bs,bj,pp in rows if bs >= 3 and bs > ss)
has_strong = sum(1 for i,c,ss,bs,bj,pp in rows if bs >= 3)
print(f"{len(rows)} entries")
print(f"  own prompt is (tied) best match : {own_best}")
print(f"  matches some OTHER prompt clearly better (>=3) : {strong_elsewhere}")
print(f"  parts strongly match SOME prompt (>=3 anywhere) : {has_strong}  "
      f"(high => lists are individually valid, just mis-keyed => re-pairing feasible)")

offs = Counter(bj - i for i,c,ss,bs,bj,pp in rows if bs >= 3 and bs > ss)
print("\noffset (best_j - i) histogram, entries matching elsewhere better (>=3):")
for off, n in offs.most_common(12):
    print(f"  {off:+5d} : {n}")
print("  (one dominant offset => mechanical shift, recoverable | flat => true shuffle)")

print(f"\n{'category':24}{'n':>4}{'self':>7}{'best':>7}{'self_ok':>9}")
bycat = {}
for i,c,ss,bs,bj,pp in rows: bycat.setdefault(c, []).append((ss, bs))
for c, v in sorted(bycat.items()):
    ok = sum(1 for s,b in v if s >= b and s > 0)
    print(f"{c:24}{len(v):>4}{mean(s for s,_ in v):>7.2f}{mean(b for _,b in v):>7.2f}{ok:>9}")

print("\nFURNITURE (expect self to dominate if this block is clean):")
for i,c,ss,bs,bj,pp in rows:
    if c == "furniture":
        tag = "" if ss >= bs else f"  <-- prompt {bj} fits better (+{bs} vs +{ss})"
        print(f"  {i} self+{ss} best+{bs}  {pp}{tag}")
