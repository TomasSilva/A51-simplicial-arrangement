"""Exact analysis of the extension A(21,2) -> A51 (standalone: Python Fractions + sympy; no simparr code).

    python simplicial-paper-new/code/construction.py

Arithmetic is exact in K = Q(s), s = sqrt(phi), s^4 = s^2 + 1 (so phi = s^2, sqrt5 = 2 s^2 - 1); every element is a
4-tuple of Fractions in the basis 1, s, s^2, s^3.  Lines are normal vectors (a line (a:b:c) is the plane
ax + by + cz = 0), the icosahedral rotation group I acts on normals, and incidences are exact zero tests.

Outputs (in ../generated/): summary.json with every number quoted in the paper, and the LaTeX tables
a21_points.tex, a51_points.tex, generic_points.tex, conditions.tex.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from fractions import Fraction as Fr
from pathlib import Path

import sympy as sp

OUT = Path(__file__).resolve().parent.parent / "generated"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------- the field Q(s), s^4 = s^2 + 1
Z4 = (Fr(0),) * 4


def el(a=0, b=0, c=0, d=0):
    return (Fr(a), Fr(b), Fr(c), Fr(d))


ONE, ZERO = el(1), Z4
S = el(0, 1)                        # sqrt(phi)
PHI = el(0, 0, 1)                   # phi = s^2
PHI_INV = el(-1, 0, 1)              # 1/phi = phi - 1
SQRT5 = el(-1, 0, 2)                # sqrt5 = 2 phi - 1
HALF = el(Fr(1, 2))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def neg(a):
    return tuple(-x for x in a)


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    out = [Fr(0)] * 7
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    out[i + j] += x * y
    # reduce with s^4 = s^2 + 1 (from the top degree down)
    for k in range(6, 3, -1):
        c = out[k]
        if c:
            out[k] = Fr(0)
            out[k - 2] += c
            out[k - 4] += c
    return tuple(out[:4])


def mulmat(a):
    """4x4 matrix of multiplication by a (columns = images of the basis)."""
    cols = [mul(a, tuple(Fr(int(i == j)) for i in range(4))) for j in range(4)]
    return [[cols[j][i] for j in range(4)] for i in range(4)]


def inv(a):
    """Inverse by solving (mult by a) x = 1 with Gaussian elimination over Q."""
    M = mulmat(a)
    aug = [row[:] + [Fr(int(i == 0))] for i, row in enumerate(M)]
    n = 4
    for c in range(n):
        p = next(r for r in range(c, n) if aug[r][c] != 0)
        aug[c], aug[p] = aug[p], aug[c]
        pv = aug[c][c]
        aug[c] = [x / pv for x in aug[c]]
        for r in range(n):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[c])]
    return tuple(aug[i][n] for i in range(n))


def is_zero(a):
    return not any(a)


def to_float(a):
    s = ((1 + 5 ** 0.5) / 2) ** 0.5
    return sum(float(c) * s ** i for i, c in enumerate(a))


def to_sympy(a):
    s = sp.sqrt((1 + sp.sqrt(5)) / 2)
    return sp.nsimplify(sum(sp.Rational(c.numerator, c.denominator) * s ** i for i, c in enumerate(a)))


# ---------------------------------------------------------------------------- vectors, lines, points
def dot(u, v):
    return add(add(mul(u[0], v[0]), mul(u[1], v[1])), mul(u[2], v[2]))


def cross(u, v):
    return (sub(mul(u[1], v[2]), mul(u[2], v[1])), sub(mul(u[2], v[0]), mul(u[0], v[2])),
            sub(mul(u[0], v[1]), mul(u[1], v[0])))


def canon(v):
    """Projective normal form: divide by the first nonzero coordinate."""
    for c in v:
        if not is_zero(c):
            ic = inv(c)
            return tuple(mul(x, ic) for x in v)
    raise ValueError("zero vector")


def matvec(g, v):
    return tuple(add(add(mul(g[3 * i], v[0]), mul(g[3 * i + 1], v[1])), mul(g[3 * i + 2], v[2])) for i in range(3))


def matmul(g, h):
    return tuple(add(add(mul(g[3 * i], h[j]), mul(g[3 * i + 1], h[3 + j])), mul(g[3 * i + 2], h[6 + j]))
                 for i in range(3) for j in range(3))


# ---------------------------------------------------------------------------- the icosahedral rotation group
Cyc = (ZERO, ZERO, ONE, ONE, ZERO, ZERO, ZERO, ONE, ZERO)       # (x, y, z) -> (z, x, y)
Half = (ONE, ZERO, ZERO, ZERO, neg(ONE), ZERO, ZERO, ZERO, neg(ONE))  # half-turn about the x-axis
R5 = tuple(mul(HALF, e) for e in (ONE, neg(PHI), PHI_INV, PHI, PHI_INV, neg(ONE), PHI_INV, ONE, PHI))
group = {(ONE, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ONE)}
frontier = list(group)
while frontier:
    new = []
    for g in frontier:
        for h in (Cyc, Half, R5):
            k = matmul(g, h)
            if k not in group:
                group.add(k)
                new.append(k)
    frontier = new
I = sorted(group)
assert len(I) == 60, len(I)


def orbit(v):
    return sorted({canon(matvec(g, v)) for g in I})


def vec(*xs):
    return tuple(x if isinstance(x, tuple) else el(x) for x in xs)


O15 = orbit(vec(1, 0, 0))                     # planes perpendicular to the 2-fold axes (mirrors of H3)
O6 = orbit(vec(0, PHI, 1))                    # planes perpendicular to the 5-fold axes
O10 = orbit(vec(0, PHI_INV, PHI))             # planes perpendicular to the 3-fold axes
assert (len(O15), len(O6), len(O10)) == (15, 6, 10)
A21 = O15 + O6
LINE_ORBIT = {}
for L in O15:
    LINE_ORBIT[L] = "15"
for L in O6:
    LINE_ORBIT[L] = "6"


def ell(t):
    return canon(vec(0, 1, t))


def arrangement(t):
    """A(t) = A(21,2) u I.(0:1:t) (lines, with orbit labels)."""
    O30 = orbit(vec(0, 1, t))
    lines = sorted(set(A21) | set(O30))
    lab = dict(LINE_ORBIT)
    for L in O30:
        lab.setdefault(L, "new")
    return lines, lab, O30


def points(lines):
    pts = {}
    for i, j in itertools.combinations(range(len(lines)), 2):
        P = canon(cross(lines[i], lines[j]))
        if P not in pts:
            pts[P] = [k for k in range(len(lines)) if is_zero(dot(lines[k], P))]
    return pts


def point_orbits(lines, lab):
    pts = points(lines)
    seen, rows = set(), []
    for P, through in sorted(pts.items(), key=lambda x: (-len(x[1]), x[0])):
        if P in seen:
            continue
        orb = {canon(matvec(g, P)) for g in I}
        seen |= orb
        c = Counter(lab[lines[k]] for k in through)
        rows.append({"size": len(orb), "mult": len(through), "through": dict(c), "rep": P})
    t = Counter()
    for r in rows:
        t[r["mult"]] += r["size"]
    n = len(lines)
    assert sum(v * m * (m - 1) // 2 for m, v in t.items()) == n * (n - 1) // 2
    delta = t[2] - 3 - sum((m - 3) * v for m, v in t.items() if m >= 4)
    return rows, dict(sorted(t.items())), delta, len(pts)


def fmt_el(a):
    """LaTeX for an element of Q(s) as (u + v phi) + (w + x phi) sqrt(phi), with s^2 = phi."""
    c0, c1, c2, c3 = a

    def part(u, v):
        if u == 0 and v == 0:
            return "0"
        out = ""
        if u:
            out = str(u)
        if v:
            coef = "" if abs(v) == 1 else str(abs(v))
            out += ("-" if v < 0 else ("+" if out else "")) + coef + r"\varphi"
        return out

    r, q = part(c0, c2), part(c1, c3)
    if q == "0":
        return r
    sq = r"\sqrt{\varphi}"
    if c3 == 0:
        st = ("" if c1 == 1 else "-" if c1 == -1 else str(c1)) + sq
    elif c1 == 0:
        st = ("" if c3 == 1 else "-" if c3 == -1 else str(c3)) + r"\varphi" + sq
    else:
        st = f"({q}){sq}"
    if r == "0":
        return st
    return r + ("" if st.startswith("-") else "+") + st


def fmt_vec(v):
    return "(" + ":".join(fmt_el(x) for x in v) + ")"


def table(rows, cols, path, caption_cols):
    lines = [r"\begin{tabular}{rr" + "r" * len(cols) + "l}", r"\toprule",
             "size & mult. & " + " & ".join(caption_cols) + r" & representative\\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['size']} & {r['mult']} & " + " & ".join(str(r["through"].get(c, 0)) for c in cols)
                     + f" & ${fmt_vec(r['rep'])}$\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / path).write_text("\n".join(lines) + "\n")


summary = {}

# ---------------------------------------------------------------------------- A(21,2)
rows21, t21, d21, f0_21 = point_orbits(A21, LINE_ORBIT)
summary["A21"] = {"n": 21, "t": t21, "Delta": d21, "f0": f0_21,
                  "orbits": [{k: v for k, v in r.items() if k != "rep"} for r in rows21]}
table(rows21, ["15", "6"], "a21_points.tex", [r"$\#\cO_{15}$", r"$\#\cO_6$"])

# ---------------------------------------------------------------------------- the pencil at P = (1:0:0)
P = vec(1, 0, 0)
pencil = {}
for name, lines in (("mirror", O15), ("O6", O6), ("O10", O10)):
    for L in lines:
        if is_zero(dot(L, P)):
            # L = (0 : b : c); pencil parameter t = c / b (t = inf for b = 0)
            b, c = L[1], L[2]
            pencil.setdefault(name, []).append("inf" if is_zero(b) else round(to_float(mul(c, inv(b))), 12))
summary["pencil_at_P"] = {k: sorted(v, key=lambda x: (x == "inf", x if x != "inf" else 0)) for k, v in pencil.items()}
# the involution t -> phi/t swaps the mirrors (0, inf) and O6 <-> O10 (1/phi <-> phi^2); fixed points t^2 = phi
assert mul(PHI_INV, el(0, 0, 0, 0)) == ZERO
assert mul(mul(PHI_INV, mul(PHI, PHI)), ONE) == PHI   # (1/phi) * phi^2 = phi
summary["involution"] = "t -> phi/t: 0 <-> inf, 1/phi <-> phi^2, fixed points t = +-sqrt(phi)"

# stabiliser of the moving line: exactly {1, half-turn about the x-axis} for generic t
stab = [g for g in I if canon(matvec(g, vec(0, 1, Fr(3, 7)))) == ell(Fr(3, 7))]
summary["stabiliser_generic"] = len(stab)

# ---------------------------------------------------------------------------- generic t and t = sqrt(phi)
for key, t in (("generic", Fr(3, 7)), ("A51", S)):
    lines, lab, O30 = arrangement(t if isinstance(t, tuple) else el(t))
    rows, tv, dl, f0 = point_orbits(lines, lab)
    summary[key] = {"n": len(lines), "orbit": len(O30), "t": tv, "Delta": dl, "f0": f0,
                    "orbits": [{k: v for k, v in r.items() if k != "rep"} for r in rows]}
    table(rows, ["15", "6", "new"], f"{key}_points.tex",
          [r"$\#\cO_{15}$", r"$\#\cO_6$", r"$\#\cO_{30}$"])
    if key == "A51":
        summary["A51"]["new_lines"] = [fmt_vec(L) for L in O30]
        # vertices on the line ell = (0:1:sqrt(phi)) and the lines through them
        L0 = ell(S)
        on = []
        for Pt, through in points(lines).items():
            if is_zero(dot(L0, Pt)):
                on.append(sorted(Counter(lab[lines[k]] for k in through).items()))
        summary["A51"]["points_on_new_line"] = sorted(Counter(tuple(x) for x in on).items())

# ---------------------------------------------------------------------------- why t^2 = phi: condition polynomials
tt = sp.Symbol("t")
phi_s = (1 + sp.sqrt(5)) / 2
K5 = sp.QQ.algebraic_field(sp.sqrt(5))


def sym_vec(v):
    return [sp.nsimplify(to_sympy(x)) for x in v]


rot = [[sym_vec(g[3 * i:3 * i + 3]) for i in range(3)] for g in I]


def sym_matvec(G, v):
    return [sp.expand(sum(G[i][j] * v[j] for j in range(3))) for i in range(3)]


base_sym = [(sym_vec(L), LINE_ORBIT[L]) for L in A21]
moving = []
seen = set()
for G in rot:
    w = sym_matvec(G, [0, 1, tt])
    key = tuple(sp.N(x.subs(tt, sp.Rational(3, 7)), 30) for x in w)
    nrm = max(abs(x) for x in key)
    # normalise sign/scale numerically for deduplication only
    k0 = next(x for x in key if abs(x) > 1e-20)
    keyn = tuple(round(float(x / k0), 12) for x in key)
    if keyn in seen:
        continue
    seen.add(keyn)
    moving.append(w)
assert len(moving) == 30
ell_t = moving[[i for i, w in enumerate(moving) if sp.simplify(w[0]) == 0 and sp.simplify(w[1] - 1) == 0
                and sp.simplify(w[2] - tt) == 0][0]] if any(sp.simplify(w[0]) == 0 and sp.simplify(w[1] - 1) == 0
                                                            and sp.simplify(w[2] - tt) == 0 for w in moving) else [0, 1, tt]
others = [(v, k) for v, k in base_sym] + [(w, "new") for w in moving if [sp.simplify(a - b) for a, b in zip(w, ell_t)] != [0, 0, 0]]
cond = Counter()
examples = defaultdict(list)
target = sp.Poly(tt ** 2 - phi_s, tt, domain=K5)
Phi = sp.Symbol("varphi")


def pretty(e):
    """a + b*phi form of an element of Q(sqrt5)[t] coefficientwise (sqrt5 = 2 phi - 1), for display."""
    e = sp.expand(e)
    poly = sp.Poly(e, tt)
    out = 0
    for (k,), c in poly.terms():
        c = sp.nsimplify(c)
        a, b = sp.Rational(0), sp.Rational(0)
        cc = sp.expand(c)
        b5 = cc.coeff(sp.sqrt(5))
        a5 = sp.expand(cc - b5 * sp.sqrt(5))
        # a5 + b5 sqrt5 = (a5 - b5) + 2 b5 phi
        a, b = a5 - b5, 2 * b5
        out += (a + b * Phi) * tt ** k
    return sp.collect(sp.expand(out), tt)


for (X, kx), (Y, ky) in itertools.combinations(others, 2):
    d = sp.Matrix([ell_t, X, Y]).det()
    p = sp.Poly(sp.expand(d), tt, domain=K5)
    if p.is_zero:
        continue
    if p.rem(target).is_zero:
        kind = tuple(sorted((kx, ky)))
        cond[kind] += 1
        examples[kind].append({"X": [str(pretty(x)) for x in X], "kx": kx, "Y": [str(pretty(y)) for y in Y], "ky": ky,
                               "det": str(sp.factor(p.as_expr(), extension=sp.sqrt(5))),
                               "size": sum(len(str(pretty(x))) for x in list(X) + list(Y))})
for k in examples:
    examples[k].sort(key=lambda e: e["size"])
    examples[k] = examples[k][:4]
summary["conditions_with_factor_t2_minus_phi"] = {"/".join(k): v for k, v in cond.items()}
summary["condition_examples"] = {"/".join(k): v for k, v in examples.items()}

# ---------------------------------------------------------------------------- the two coincidences, exactly
# (a) four lines of the orbit through one point of the mirror x = 0
g1 = tuple(mul(HALF, e) for e in (PHI_INV, ONE, PHI, neg(ONE), PHI, neg(PHI_INV), neg(PHI), neg(PHI_INV), ONE))
g2 = tuple(mul(HALF, e) for e in (PHI_INV, neg(ONE), PHI, ONE, PHI, PHI_INV, neg(PHI), PHI_INV, ONE))
assert g1 in group and g2 in group, "g1, g2 must be rotations of the icosahedron"
eta = Half
kappa_inv = (ZERO, ONE, ZERO, ZERO, ZERO, ONE, ONE, ZERO, ZERO)   # (x, y, z) -> (y, z, x)
assert kappa_inv in group


def lines_at(t):
    L = vec(0, 1, t)
    L1p, L2p = matvec(g1, L), matvec(g2, L)
    return L, L1p, matvec(eta, L1p), L2p, matvec(eta, L2p), matvec(eta, matvec(kappa_inv, L))


checks = {}
for name, t in (("generic 3/7", el(Fr(3, 7))), ("sqrt(phi)", S)):
    L, L1p, L1m, L2p, L2m, Lb = lines_at(t)
    m = vec(1, 0, 0)                                    # the mirror x = 0
    o = vec(1, 0, PHI)                                  # a line of O6
    Q1, Q2 = canon(cross(L1p, L1m)), canon(cross(L2p, L2m))
    checks[name] = {
        "L1+ - L1- meet on x=0": is_zero(dot(m, Q1)),
        "L2+ - L2- meet on x=0": is_zero(dot(m, Q2)),
        "the two points coincide": Q1 == Q2,
        "l_t, eta kappa^-1 l_t and (1:0:phi) concurrent": is_zero(dot(o, cross(L, Lb))),
        "(1:0:phi) in O6": canon(o) in O6,
    }
assert all(checks["sqrt(phi)"].values())
assert checks["generic 3/7"]["L1+ - L1- meet on x=0"] and not checks["generic 3/7"]["the two points coincide"]
assert not checks["generic 3/7"]["l_t, eta kappa^-1 l_t and (1:0:phi) concurrent"]
summary["coincidence_checks"] = checks
# symbolic forms (for the paper): L1+, L2+ and the meeting points as functions of t
ph = sp.Symbol("varphi", positive=True)


def sym_line(g):
    def c(e):
        return sp.nsimplify(to_sympy(e)).subs(sp.sqrt(5), 2 * ph - 1)
    return [sp.simplify(c(g[3 * i + 1]) + tt * c(g[3 * i + 2])) for i in range(3)]


summary["L1+"] = [str(x) for x in sym_line(g1)]
summary["L2+"] = [str(x) for x in sym_line(g2)]

# ---------------------------------------------------------------------------- data file, line table, figure
lines51, lab51, O30 = arrangement(S)
(OUT.parent / "data").mkdir(exist_ok=True)
(OUT.parent / "data" / "a51.json").write_text(json.dumps({
    "description": "A51 = O15 u O6 u I.(0:1:sqrt(phi)); lines (a:b:c) as normals; coordinates in the power basis "
                   "1, s, s^2, s^3 of s = sqrt(phi), s^4 = s^2 + 1",
    "number_field": {"poly": [-1, 0, -1, 0, 1], "interval": ["127/100", "128/100"]},
    "orbits": {"O15": [[[str(c) for c in x] for x in L] for L in O15],
               "O6": [[[str(c) for c in x] for x in L] for L in O6],
               "O30": [[[str(c) for c in x] for x in L] for L in O30]},
    "lines": [[[str(c) for c in x] for x in L] for L in lines51]}, indent=1))
rows_t = [r"\begin{longtable}{ll}", r"\toprule"]
for k in range(0, len(O30), 2):
    cells = [f"${fmt_vec(L)}$" for L in O30[k:k + 2]]
    rows_t.append(" & ".join(cells + [""] * (2 - len(cells))) + r"\\")
rows_t += [r"\bottomrule", r"\end{longtable}"]
(OUT / "new_lines.tex").write_text("\n".join(rows_t) + "\n")

import math  # noqa: E402

phf = (1 + 5 ** 0.5) / 2


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def crs(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def dt(a, b):
    return sum(x * y for x, y in zip(a, b))


w = unit([0, phf, 1])                      # view along a 5-fold axis
ax = unit(crs(w, [1, 0, 0]))
bx = crs(w, ax)


def tikz(lines, style, shift=0.0):
    cmds = [f"\\draw[gray, line width=0.4pt] ({shift},0) circle (1);"]
    for L in lines:
        n = unit([to_float(c) for c in L])
        mv = [dt(ax, n), dt(bx, n), dt(w, n)]
        hx, hy = -mv[1], mv[0]
        hn = math.hypot(hx, hy)
        st = style(L)
        if hn < 1e-12:
            cmds.append(f"\\draw[{st}] ({shift},0) circle (1);")
            continue
        u = [hx / hn, hy / hn, 0.0]
        v = crs(mv, u)
        if v[2] < 0:
            v = [-x for x in v]
        ang = math.degrees(math.atan2(u[1], u[0]))
        side = 1 if (-u[1] * v[0] + u[0] * v[1]) >= 0 else -1
        cmds.append(f"\\draw[{st}, shift={{({shift},0)}}, rotate={ang:.4f}] (1,0) arc[start angle=0, "
                    f"end angle={180 * side}, x radius=1, y radius={abs(mv[2]):.5f}];")
    return cmds


sty = {"15": "line width=0.8pt, black", "6": "line width=0.8pt, blue!70!black", "new": "line width=0.4pt, red!75!black"}
order = {"new": 0, "6": 1, "15": 2}
cmds = tikz(sorted(A21, key=lambda L: order[LINE_ORBIT[L]]), lambda L: sty[LINE_ORBIT[L]], 0.0)
cmds += tikz(sorted(lines51, key=lambda L: order[lab51[L]]), lambda L: sty[lab51[L]], 2.35)
(OUT / "figure.tex").write_text("\\begin{tikzpicture}[scale=2.6]\n" + "\n".join(cmds) + "\n\\end{tikzpicture}\n")

(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps({k: summary[k] for k in ("coincidence_checks", "L1+", "L2+", "pencil_at_P", "stabiliser_generic")},
                 indent=1, default=str))
print("A51:", {k: v for k, v in summary["A51"].items() if k != "new_lines"})
