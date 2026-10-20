"""Independent exact re-check of an arrangement given as JSON (no simparr code).

    python docs/results/candidates/vet_exact.py candidate.json

Input JSON: {"number_field": {"poly": [c0, ..., cd] (low -> high, monic, integers as strings or ints),
             "interval": [lo, hi] (rationals as strings isolating the real root)} or "QQ",
             "lines": [[coord0, coord1, coord2], ...]} where each coordinate is a list of d rational coefficients in
the power basis 1, a, ..., a^(d-1) of the generator a (a single rational for QQ).

Arithmetic: Python Fractions in Q[x]/(g); g is checked irreducible with sympy, and the interval is checked to
isolate a real root (sympy real-root isolation). Incidence is exact. The t-vector, the pair identity and
Delta = t2 - 3 - sum_{r>=4} (r-3) t_r are computed from exact incidences. Delta depends only on the intersection
lattice, so by Melchior, Delta = 0 means the real arrangement (in this, or any other real embedding) is simplicial.
"""
import itertools
import json
import sys
import time
from collections import Counter
from fractions import Fraction

import sympy as sp


def load(path):
    d = json.load(open(path))
    nf = d["number_field"]
    if nf in ("QQ", None):
        g = None
    else:
        g = [Fraction(c) for c in nf["poly"]]
        lo, hi = (Fraction(nf["interval"][0]), Fraction(nf["interval"][1]))
        x = sp.Symbol("x")
        gp = sp.Poly(sum(sp.Rational(c.numerator, c.denominator) * x ** i for i, c in enumerate(g)), x)
        assert gp.is_irreducible, "defining polynomial is reducible"
        inside = [r for r in gp.real_roots() if sp.Rational(lo.numerator, lo.denominator) < r <=
                  sp.Rational(hi.numerator, hi.denominator)]
        assert len(inside) == 1, f"interval isolates {len(inside)} real roots"
    lines = []
    for L in d["lines"]:
        if g is None:
            lines.append(tuple([Fraction(c if not isinstance(c, list) else c[0])] for c in L))
        else:
            lines.append(tuple([Fraction(c) for c in coord] for coord in L))
    return g, lines


class Field:
    def __init__(self, g):
        self.g = g  # monic, low -> high
        self.d = 1 if g is None else len(g) - 1

    def norm(self, a):
        a = list(a) + [Fraction(0)] * max(0, self.d - len(a))
        if self.g is None:
            return a[:1]
        a = list(a)
        while len(a) > self.d:
            c = a.pop()
            if c:
                for i in range(self.d):
                    a[len(a) - self.d + i] -= c * self.g[i]
        return a

    def add(self, a, b):
        return [x + y for x, y in zip(a, b)]

    def sub(self, a, b):
        return [x - y for x, y in zip(a, b)]

    def mul(self, a, b):
        out = [Fraction(0)] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    if y:
                        out[i + j] += x * y
        return self.norm(out)

    def is_zero(self, a):
        return not any(a)

    def inv(self, a):
        """Inverse via the extended Euclidean algorithm in Q[x] (g irreducible)."""
        if self.g is None:
            return [1 / a[0]]

        def trim(p):
            p = list(p)
            while p and p[-1] == 0:
                p.pop()
            return p

        def pdivmod(p, q):
            p, q = trim(p), trim(q)
            quo = [Fraction(0)] * max(1, len(p) - len(q) + 1)
            while len(p) >= len(q) and p:
                c = p[-1] / q[-1]
                k = len(p) - len(q)
                quo[k] = c
                for i, y in enumerate(q):
                    p[i + k] -= c * y
                p = trim(p)
            return quo, p

        def pmul(p, q):
            if not p or not q:
                return []
            out = [Fraction(0)] * (len(p) + len(q) - 1)
            for i, x in enumerate(p):
                for j, y in enumerate(q):
                    out[i + j] += x * y
            return out

        def psub(p, q):
            n = max(len(p), len(q))
            return trim([(p[i] if i < len(p) else 0) - (q[i] if i < len(q) else 0) for i in range(n)])

        r0, r1 = trim(self.g), trim(a)
        s0, s1 = [], [Fraction(1)]
        while r1:
            q, r = pdivmod(r0, r1)
            r0, r1 = r1, r
            s0, s1 = s1, psub(s0, pmul(q, s1))
        assert len(r0) == 1, "element not invertible"
        return self.norm([c / r0[0] for c in s0])


def main(path):
    t0 = time.time()
    g, lines = load(path)
    K = Field(g)
    lines = [tuple(K.norm(c) for c in L) for L in lines]

    def cross(u, v):
        return (K.sub(K.mul(u[1], v[2]), K.mul(u[2], v[1])), K.sub(K.mul(u[2], v[0]), K.mul(u[0], v[2])),
                K.sub(K.mul(u[0], v[1]), K.mul(u[1], v[0])))

    def dot(u, v):
        return K.add(K.add(K.mul(u[0], v[0]), K.mul(u[1], v[1])), K.mul(u[2], v[2]))

    def key(v):
        for c in v:
            if not K.is_zero(c):
                ic = K.inv(c)
                return tuple(tuple(K.mul(x, ic)) for x in v)
        raise ValueError("zero vector: two lines coincide")

    n = len(lines)
    assert len({key(L) for L in lines}) == n, "repeated line"
    points = {}
    for i, j in itertools.combinations(range(n), 2):
        p = key(cross(lines[i], lines[j]))
        if p in points:
            continue
        pv = tuple(list(c) for c in p)
        points[p] = [k for k in range(n) if K.is_zero(dot(lines[k], pv))]
    t = Counter(len(v) for v in points.values())
    pairs = sum(len(v) * (len(v) - 1) // 2 for v in points.values())
    delta = t[2] - 3 - sum((r - 3) * c for r, c in t.items() if r >= 4)
    rv = Counter(sum(1 for v in points.values() if k in v) for k in range(n))
    out = {"n": n, "f0": len(points), "t_vector": dict(sorted(t.items())), "r_vector": dict(sorted(rv.items())),
           "pair_identity": pairs == n * (n - 1) // 2, "Delta": delta,
           "near_pencil": max(t) >= n - 1, "simplicial": delta == 0 and max(t) < n - 1 and pairs == n * (n - 1) // 2,
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
