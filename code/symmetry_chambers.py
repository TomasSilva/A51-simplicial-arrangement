"""Projective symmetry group and chambers of A51 (uses the simparr package from the repository root).

    .venv/bin/python simplicial-paper-new/code/symmetry_chambers.py

Reads ../data/a51.json.  The automorphism search is exact (frame enumeration mod p, exact matrices, exact check on
every line); the chamber enumeration is done by simparr's independent verifier.
"""
import json
from fractions import Fraction
from pathlib import Path

from simparr.arith.nf import NumberField
from simparr.discovery.catalogue_ext import automorphisms
from simparr.discovery.symmetric import FieldOps
from simparr.search.fields import verify_any

DATA = Path(__file__).resolve().parent.parent / "data" / "a51.json"
d = json.loads(DATA.read_text())
F = NumberField("Q(sqrt(phi))", [int(c) for c in d["number_field"]["poly"]],
                tuple(Fraction(x) for x in d["number_field"]["interval"]))
fo = FieldOps(F)


def line(L):
    """Exact line over F: integer coordinates after clearing denominators, then canonical form."""
    coords = [[Fraction(c) for c in x] for x in L]
    den = 1
    for x in coords:
        for c in x:
            den = den * c.denominator // __import__("math").gcd(den, c.denominator)
    flat = tuple(int(c * den) for x in coords for c in x)
    return fo.canonical(flat)


lines = [line(L) for L in d["lines"]]
perms, mats = automorphisms(fo, lines)
rep = verify_any(F, lines, None, check_chambers=True)
out = {"n": len(lines), "automorphism_group_order": len(perms), "status": rep["status"],
       "t_vector": rep["t_vector"], "chambers": rep.get("chambers")}
print(json.dumps(out, indent=1, default=str))
(Path(__file__).resolve().parent.parent / "generated" / "symmetry_chambers.json").write_text(
    json.dumps(out, indent=1, default=str))
