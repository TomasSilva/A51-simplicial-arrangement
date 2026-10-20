"""Every special member of the family A(t) = A(21,2) u I.(0:1:t), with its defect computed exactly.

    .venv/bin/python simplicial-paper-new/code/family_exact.py

Uses the simparr package (exact number fields and the independent verifier) from the repository root.

By I-invariance, the incidence lattice of A(t) can change only where det((0:1:t), X(t), Y(t)) = 0 for two other
lines X, Y of A(t); these are polynomials of degree <= 3 in t.  Their real roots (and t = inf) are the special
parameters.  Every other t has the lattice of a generic t (checked at t = 3/7).  For each special t the script
refines the root to 260 digits, recognises its minimal polynomial, rebuilds A(t) exactly over Q(sqrt5)(t) and
computes the t-vector and Delta = t2 - 3 - sum_{r>=4} (r-3) t_r with exact incidences.  Output:
../generated/family_table.tex and ../generated/family.json.
"""
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from simparr.discovery import orbitscan as OS
from simparr.discovery.catalogue_ext import build_explicit
from simparr.discovery.symmetric import FieldOps, base_lines, make_group, mirrors
from simparr.search.fields import resolve_field, verify_any

OUT = Path(__file__).resolve().parent.parent / "generated"
F = resolve_field("Q(sqrt5)")
fo = FieldOps(F)
G = make_group(fo, "I")
a31 = base_lines(fo, G, "A31")
O15 = mirrors(G)
O6 = sorted(L for L in a31 if len(G.orbit_exact(L)) == 6)
base = sorted(set(O15) | set(O6))
assert len(base) == 21
z, o = fo.zero, fo.one
P, Q = z + o + z, z + z + o          # L(t) = (0 : 1 : t)

rows = OS.orbit_rows(G, fo, P, Q)
base_f = [np.array(OS.vec(fo, L)) for L in base]
cands = sorted(OS.condition_roots(base_f, rows))
special = []
for t, pair in cands:
    if abs(t) > 1e9:
        continue
    if special and abs(t - special[-1][0]) <= 1e-9 * max(1.0, abs(t)):
        continue
    special.append((t, pair))


def exact_member(Pv, Qv, th, poly):
    b = next(build_explicit(F, base, list(G.mats), Pv, Qv, th, poly))
    fo_k = b["fo"]
    orb = {fo_k.canonical(fo_k.line_times(b["L"], g)) for g in b["mats"]}
    lines = sorted(set(b["base"]) | orb)
    rep = verify_any(b["F"], lines, {"min_lines": 6, "require_essential": True, "near_pencil_offset": 2},
                     check_chambers=False)
    tv = {int(k): v for k, v in rep["t_vector"].items()}
    delta = tv.get(2, 0) - 3 - sum((r - 3) * c for r, c in tv.items() if r >= 4)
    return len(lines), len(orb), dict(sorted(tv.items())), delta, rep["status"]


results = []
for t, pair in special:
    th = OS.refine(G, fo, base, P, Q, t, pair, dps=260)
    poly = OS.recognise(th)
    n, osz, tv, delta, status = exact_member(P, Q, th, poly)
    results.append({"t": float(th), "minpoly_low_to_high": poly, "n": n, "orbit": osz, "t_vector": tv,
                    "Delta": delta, "status": status})
# t = inf: the line Q itself, and a generic parameter
mp.mp.dps = 60
for label, (Pv, Qv, th, poly) in (("inf", (Q, P, mp.mpf(0), [0, 1])), ("generic 3/7", (P, Q, mp.mpf(3) / 7, [-3, 7]))):
    n, osz, tv, delta, status = exact_member(Pv, Qv, th, poly)
    results.append({"t": label, "minpoly_low_to_high": poly, "n": n, "orbit": osz, "t_vector": tv, "Delta": delta,
                    "status": status})

(OUT / "family.json").write_text(json.dumps(results, indent=1, default=str))

# table: one row per pair {t, -t} (the half-turn about the y-axis maps (0:1:t) to (0:1:-t))
x = __import__("sympy").Symbol("t")
import sympy as sp  # noqa: E402

rows_tex = []
seen = set()
for r in results:
    if isinstance(r["t"], str):
        continue
    if r["t"] < -1e-12:
        continue
    key = round(r["t"], 9)
    if key in seen:
        continue
    seen.add(key)
    cs = r["minpoly_low_to_high"]
    sgn = 1 if cs[-1] > 0 else -1
    mpoly = sp.Poly([sgn * c for c in reversed(cs)], x).as_expr()
    tv = ", ".join(f"{k}:{v}" for k, v in r["t_vector"].items())
    rows_tex.append((r["t"], f"${r['t']:.6f}$ & ${sp.latex(mpoly)}$ & {r['orbit']} & {r['n']} & $\\{{{tv}\\}}$ & "
                             f"{r['Delta']}\\\\"))
rows_tex.sort()
gen = next(r for r in results if r["t"] == "generic 3/7")
lines = [r"\begin{tabular}{rlrrlr}", r"\toprule",
         r"$t$ & minimal polynomial of $t$ & orbit & $n$ & $t$-vector & $\Delta$\\", r"\midrule"]
lines += [s for _, s in rows_tex]
tvg = ", ".join(f"{k}:{v}" for k, v in gen["t_vector"].items())
lines.append(r"\midrule")
lines.append(f"generic & & {gen['orbit']} & {gen['n']} & $\\{{{tvg}\\}}$ & {gen['Delta']}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "family_table.tex").write_text("\n".join(lines) + "\n")
print(len(results), "members;", "simplicial:", [(r["t"], r["n"]) for r in results if r["Delta"] == 0])
print("min Delta among non-simplicial:", min(r["Delta"] for r in results if r["Delta"] > 0))
for r in sorted(results, key=lambda r: (isinstance(r["t"], str), r["t"] if not isinstance(r["t"], str) else 0)):
    print(r["t"] if isinstance(r["t"], str) else f"{r['t']:+.6f}", r["minpoly_low_to_high"], r["orbit"], r["n"],
          r["t_vector"], r["Delta"])
