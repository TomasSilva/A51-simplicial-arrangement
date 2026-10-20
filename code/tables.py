"""Point-type comparison table (generic member of the family against t = sqrt(phi)) from ../generated/summary.json.

    python simplicial-paper-new/code/tables.py      (after construction.py)
"""
import json
from collections import Counter
from pathlib import Path

GEN = Path(__file__).resolve().parent.parent / "generated"
s = json.loads((GEN / "summary.json").read_text())

NAMES = {"15": "mirror", "6": r"$\cO_6$", "new": "new"}


def types(key):
    c = Counter()
    for r in s[key]["orbits"]:
        comp = tuple(sorted((k, v) for k, v in r["through"].items()))
        c[(r["mult"], comp)] += r["size"]
    return c


gen, a51 = types("generic"), types("A51")
keys = sorted(set(gen) | set(a51), key=lambda k: (-k[0], [(-v, x) for x, v in k[1]]))


def describe(comp):
    parts = []
    for k in ("15", "6", "new"):
        v = dict(comp).get(k, 0)
        if v:
            parts.append(f"{v} {NAMES[k]}" + ("s" if v > 1 and k == "15" else ""))
    return " + ".join(parts)


rows = [r"\begin{tabular}{lrrr}", r"\toprule", r"lines through the point & mult. & generic $t$ & $t=\sqrt\varphi$\\",
        r"\midrule"]
for k in keys:
    rows.append(f"{describe(k[1])} & {k[0]} & {gen.get(k, 0)} & {a51.get(k, 0)}\\\\")
rows.append(r"\midrule")
rows.append(f"all vertices & & {s['generic']['f0']} & {s['A51']['f0']}\\\\")
rows.append(f"$\\Delta$ & & {s['generic']['Delta']} & {s['A51']['Delta']}\\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
(GEN / "types_table.tex").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
