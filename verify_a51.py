#!/usr/bin/env python3
"""Reproduce the real simplicial 51-line arrangement using exact arithmetic.

Requires Python 3.9+ and its standard library only. No input files, installation,
network connection, or floating-point arithmetic are needed. The default run
writes nothing. Examples:

    python3 verify_a51.py
    python3 verify_a51.py --self-test
    python3 verify_a51.py --export-json a51_certificate.json

Every field element is a tuple (a0,a1,a2,a3) of fractions, representing
a0+a1*s+a2*s^2+a3*s^3, where s^4=s^2+1 and 5/4<s<13/10.
Thus s=sqrt((1+sqrt(5))/2), and phi=s^2. The defining polynomial is
irreducible over Q: phi has norm -1 in Q(sqrt(5)), so it cannot be a
square in that quadratic field. Equality in this field is coefficientwise.

The three displayed rotations have orders 3, 2, and 5, preserve the twelve
icosahedral vertices, and generate a group of order 60. Acting on line
normals, they generate three disjoint projective orbits of sizes 15, 6, 30.
Because the matrices are orthogonal, the same matrices act on points and
on line normals. The line with normal (a,b,c) has equation ax+by+cz=0.

The program enumerates all pairs of lines and independently checks every
line/vertex incidence. It also verifies all twelve point orbits in the
manuscript's incidence table, including their sizes and incidence profiles.
For an essential real projective line arrangement,
Euler's identity gives f2=1-f0+f1, and simpliciality is equivalent to
3*f2=2*f1, or to equality in Melchior's inequality. Irreducibility is
checked using the rank-three criterion: a reducible essential arrangement
is a near-pencil, with n-1 lines through a point.

The optional JSON certificate uses one-based line and vertex identifiers.
All coordinate entries are four rational strings in ascending powers of s.
The output is deterministic: lexicographically sorted exact coordinates set
the identifiers, and no timestamps or elapsed times are stored in the file.
The checks below remain active under python -O.
"""

import argparse
from collections import Counter, deque
from fractions import Fraction as Q
from functools import lru_cache
from itertools import combinations, product
import json
from pathlib import Path
import sys
import time


ZERO = (Q(0),) * 4
ONE = (Q(1), Q(0), Q(0), Q(0))
S = (Q(0), Q(1), Q(0), Q(0))
PHI = (Q(0), Q(0), Q(1), Q(0))


def require(condition, message):
    """Check a mathematical claim, even if Python optimization is enabled."""
    if not condition:
        raise ValueError("Verification failed: " + message)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def neg(a):
    return tuple(-x for x in a)


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


@lru_cache(maxsize=100000)
def mul(a, b):
    if a == ZERO or b == ZERO:
        return ZERO
    if a == ONE:
        return b
    if b == ONE:
        return a
    coeff = [Q(0)] * 7
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    coeff[i + j] += x * y
    # s^k = s^(k-2) + s^(k-4), for k >= 4.
    for k in range(6, 3, -1):
        coeff[k - 2] += coeff[k]
        coeff[k - 4] += coeff[k]
    return tuple(coeff[:4])


@lru_cache(maxsize=10000)
def inv(a):
    if a == ZERO:
        raise ZeroDivisionError("Cannot invert the zero field element")
    if not any(a[1:]):
        return (1 / a[0], Q(0), Q(0), Q(0))
    basis = [tuple(Q(i == j) for i in range(4)) for j in range(4)]
    cols = [mul(a, e) for e in basis]
    rows = [[cols[j][i] for j in range(4)] + [Q(i == 0)]
            for i in range(4)]
    for col in range(4):
        pivot = next((i for i in range(col, 4) if rows[i][col]), None)
        require(pivot is not None, "multiplication by a nonzero element is invertible")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        factor = rows[col][col]
        rows[col] = [x / factor for x in rows[col]]
        for i in range(4):
            if i != col and rows[i][col]:
                factor = rows[i][col]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[col])]
    result = tuple(row[4] for row in rows)
    require(mul(a, result) == ONE, "computed field inverse")
    return result


def normalize(v):
    """Unique projective representative: its first nonzero coordinate is 1."""
    first = next((a for a in v if a != ZERO), None)
    if first is None:
        raise ValueError("A zero vector has no projective normalization")
    scale = inv(first)
    return tuple(mul(a, scale) for a in v)


def dot(a, b):
    return add(add(mul(a[0], b[0]), mul(a[1], b[1])), mul(a[2], b[2]))


def cross(a, b):
    return tuple(sub(mul(a[(i + 1) % 3], b[(i + 2) % 3]),
                     mul(a[(i + 2) % 3], b[(i + 1) % 3])) for i in range(3))


def transpose(matrix):
    return tuple(zip(*matrix))


def matvec(matrix, v):
    return tuple(dot(row, v) for row in matrix)


def matmul(a, b):
    return tuple(tuple(dot(row, col) for col in transpose(b)) for row in a)


IDENTITY = ((ONE, ZERO, ZERO), (ZERO, ONE, ZERO), (ZERO, ZERO, ONE))
PHI_INV = sub(PHI, ONE)
HALF = (Q(1, 2), Q(0), Q(0), Q(0))
GENERATORS = (
    ((ZERO, ZERO, ONE), (ONE, ZERO, ZERO), (ZERO, ONE, ZERO)),
    ((ONE, ZERO, ZERO), (ZERO, neg(ONE), ZERO), (ZERO, ZERO, neg(ONE))),
    tuple(tuple(mul(HALF, x) for x in row) for row in (
        (ONE, neg(PHI), PHI_INV),
        (PHI, PHI_INV, neg(ONE)),
        (PHI_INV, ONE, PHI),
    )),
)
SEEDS = ((ONE, ZERO, ZERO), (ZERO, PHI, ONE), (ZERO, ONE, S))
ORBIT_NAMES = ("O15", "O6", "O30")
GENERATOR_NAMES = ("kappa", "eta", "rho")
GENERATOR_ORDERS = (3, 2, 5)


def icosahedral_vertices():
    """The twelve vectors (0,+/-phi,+/-1) and cyclic permutations."""
    vertices = set()
    for epsilon, eta in product((ONE, neg(ONE)), repeat=2):
        v = (ZERO, mul(epsilon, PHI), eta)
        vertices.update(v[i:] + v[:i] for i in range(3))
    require(len(vertices) == 12, "twelve distinct icosahedral vertices")
    return vertices


def verify_field():
    """Check field conventions and the selected real root with rational tests.

    The polynomial is strictly increasing on (1,infinity); the two endpoint
    signs therefore isolate its unique positive root there. Irreducibility
    follows from the norm argument documented in this module's introduction.
    """
    lo, hi = Q(5, 4), Q(13, 10)
    require(lo**4 - lo**2 - 1 < 0 < hi**4 - hi**2 - 1,
            "isolating interval for the positive real root")
    require(mul(S, S) == PHI and mul(PHI, PHI) == add(PHI, ONE),
            "s^2=phi and s^4=s^2+1")
    require(mul(PHI, PHI_INV) == ONE, "phi^(-1)=phi-1")


def generate_group():
    vertices = icosahedral_vertices()
    for name, matrix, order in zip(GENERATOR_NAMES, GENERATORS, GENERATOR_ORDERS):
        require(matmul(matrix, transpose(matrix)) == IDENTITY,
                name + " is orthogonal")
        require(dot(matrix[0], cross(matrix[1], matrix[2])) == ONE,
                name + " has determinant +1")
        require({matvec(matrix, v) for v in vertices} == vertices,
                name + " preserves the twelve icosahedral vertices")
        power = IDENTITY
        for exponent in range(1, order + 1):
            power = matmul(matrix, power)
            require((power == IDENTITY) == (exponent == order),
                    name + " has exact order " + str(order))
    found = {IDENTITY}
    todo = deque([IDENTITY])
    while todo:
        current = todo.popleft()
        for generator in GENERATORS:
            new = matmul(generator, current)
            if new not in found:
                found.add(new)
                todo.append(new)
        require(len(found) <= 60, "rotation group has at most 60 elements")
    require(len(found) == 60, "rotation group has order 60")
    require(all({matvec(matrix, v) for v in vertices} == vertices for matrix in found),
            "all sixty group elements preserve the twelve icosahedral vertices")
    return sorted(found)


def generate_lines(group):
    orbits = [sorted({normalize(matvec(g, seed)) for g in group}) for seed in SEEDS]
    require([len(orbit) for orbit in orbits] == [15, 6, 30], "projective orbit sizes")
    lines = [line for orbit in orbits for line in orbit]
    require(len(lines) == len(set(lines)) == 51, "three disjoint orbits, 51 lines")
    return lines, orbits


def compact_line_orbits():
    """Independent closed formulas for all normals displayed in the paper.

    cyc means all three cyclic coordinate permutations, and the signs are
    independent. The SAME delta occurs in both parenthesized coordinates:
      O15 = cyc(1,0,0) union cyc(1,epsilon*phi,eta*(phi-1));
      O6  = cyc(0,phi,epsilon);
      O30 = cyc(0,1,epsilon*s) union
            cyc(1,epsilon*(1+delta*s),eta*(1+delta*(phi-1)*s)).
    """
    def cyclic(v):
        return {normalize(v[i:] + v[:i]) for i in range(3)}

    signs = (ONE, neg(ONE))
    o15 = cyclic((ONE, ZERO, ZERO))
    for epsilon, eta in product(signs, repeat=2):
        o15.update(cyclic((ONE, mul(epsilon, PHI), mul(eta, PHI_INV))))
    o6 = set()
    o30 = set()
    for epsilon in signs:
        o6.update(cyclic((ZERO, PHI, epsilon)))
        o30.update(cyclic((ZERO, ONE, mul(epsilon, S))))
    for epsilon, eta, delta in product(signs, repeat=3):
        o30.update(cyclic((ONE, mul(epsilon, add(ONE, mul(delta, S))),
                           mul(eta, add(ONE, mul(delta, mul(PHI_INV, S)))))))
    return [sorted(o15), sorted(o6), sorted(o30)]


def verify_point_orbits(group, lines, line_orbits, points):
    """Check the twelve-row certificate without relying on its completeness.

    For each displayed representative, enumerate its projective orbit and
    compare every point's incidence profile with the independently computed
    lattice. Finally verify that these orbits partition the whole lattice.
    Profiles count incident lines in O15, O6, O30, in that order.
    """
    s3 = mul(PHI, S)
    phi_inv_s = mul(PHI_INV, S)
    rows = [
        (15, (2, 2, 2), (ZERO, ZERO, ONE)),
        (6, (5, 0, 0), (ZERO, ONE, PHI_INV)),
        (30, (1, 0, 4), (ZERO, ONE, neg(S))),
        (30, (1, 0, 2), (ZERO, ONE, sub(phi_inv_s, ONE))),
        (10, (3, 0, 0), (ZERO, ONE, neg(add(ONE, PHI)))),
        (30, (1, 0, 2), (ZERO, ONE, neg(add(ONE, phi_inv_s)))),
        (30, (1, 0, 2), (ZERO, ONE, neg(add(PHI, s3)))),
        (30, (1, 0, 2), (ZERO, ONE, sub(s3, PHI))),
        (60, (0, 1, 2), (PHI, S, neg(ONE))),
        (30, (1, 0, 1), (ZERO, ONE, phi_inv_s)),
        (30, (1, 1, 0), (ZERO, ONE, neg(PHI))),
        (60, (0, 0, 2), (ONE, neg(s3), s3)),
    ]
    line_type = {line: j for j, orbit in enumerate(line_orbits) for line in orbit}
    vertex_ids = {point: i + 1 for i, point in enumerate(sorted(points))}
    covered = set()
    certificate = []
    for size, profile, representative in rows:
        orbit = {normalize(matvec(g, representative)) for g in group}
        require(len(orbit) == size, "size of a displayed point orbit")
        require(not covered.intersection(orbit), "displayed point orbits are disjoint")
        for point in orbit:
            require(point in points, "displayed orbit consists of intersection points")
            actual = tuple(sum(line_type[lines[i]] == j for i in points[point])
                           for j in range(3))
            require(actual == profile, "incidence profile of every point in an orbit")
        covered.update(orbit)
        certificate.append({
            "id": len(certificate) + 1,
            "size": size,
            "multiplicity": sum(profile),
            "incidence_profile_O15_O6_O30": list(profile),
            "representative": coefficient_strings(representative),
            "vertex_ids": sorted(vertex_ids[point] for point in orbit),
        })
    require(covered == set(points), "displayed point orbits exhaust the intersection lattice")
    return certificate


def verify_special_incidences(group, line_orbits):
    """Check the two explicit coincidences explaining the choice s^2=phi."""
    def half_matrix(rows):
        return tuple(tuple(mul(HALF, a) for a in row) for row in rows)

    g1 = half_matrix(((PHI_INV, ONE, PHI),
                      (neg(ONE), PHI, neg(PHI_INV)),
                      (neg(PHI), neg(PHI_INV), ONE)))
    g2 = half_matrix(((PHI_INV, neg(ONE), PHI),
                      (ONE, PHI, PHI_INV),
                      (neg(PHI), PHI_INV, ONE)))
    require(g1 in group and g2 in group, "displayed rotations g1 and g2 belong to I")
    seed = SEEDS[2]
    half_turn = GENERATORS[1]
    four_new_lines = {
        normalize(matvec(g, seed))
        for g in (g1, matmul(half_turn, g1), g2, matmul(half_turn, g2))
    }
    p5 = (ZERO, ONE, neg(S))
    require(len(four_new_lines) == 4 and four_new_lines.issubset(set(line_orbits[2]))
            and all(dot(line, p5) == ZERO for line in four_new_lines),
            "four distinct new lines concur with the mirror x=0 at (0:1:-s)")
    kappa = GENERATORS[0]
    second_new_line = matvec(matmul(half_turn, transpose(kappa)), seed)
    old_line = matvec(kappa, SEEDS[1])
    p3 = (PHI, S, neg(ONE))
    require(normalize(second_new_line) in line_orbits[2]
            and normalize(old_line) in line_orbits[1]
            and all(dot(line, p3) == ZERO for line in (seed, second_new_line, old_line)),
            "two new lines and an O6 line concur at (phi:s:-1)")


def analyze_lines(lines, check_all_incidences=True):
    """Compute the full exact lattice and rank-three numerical invariants.

    ``points`` maps normalized coordinate triples to zero-based incident line
    indices. ``counts[i]`` is the number of distinct vertices on line i.
    ``check_all_incidences=False`` only omits the additional all-dot-products
    cross-check; it does not change the exact pair enumeration.
    """
    lines = [normalize(line) for line in lines]
    n = len(lines)
    require(n == len(set(lines)), "distinct projective line normals")
    points = {}
    for i, j in combinations(range(n), 2):
        point = normalize(cross(lines[i], lines[j]))
        points.setdefault(point, set()).update((i, j))
    t = Counter(len(incident) for incident in points.values())
    require(sum(k * (k - 1) // 2 * number for k, number in t.items())
            == n * (n - 1) // 2, "every unordered line pair occurs exactly once")
    counts = [0] * n
    for point, incident in points.items():
        if check_all_incidences:
            actual = {i for i, line in enumerate(lines) if dot(point, line) == ZERO}
            require(actual == incident, "pair enumeration agrees with every dot-product incidence")
        for i in incident:
            counts[i] += 1
    r = Counter(counts)
    rank_three = any(dot(cross(lines[i], lines[j]), lines[k]) != ZERO
                     for i, j, k in combinations(range(n), 3))
    f0 = len(points)
    f1 = sum(k * number for k, number in t.items())
    f2 = 1 - f0 + f1
    delta = t[2] - 3 - sum((k - 3) * number for k, number in t.items() if k >= 4)
    # chi(t)=t^3-n*t^2+b2*t+(n-1-b2) for an essential central rank-3 arrangement.
    b2 = sum((k - 1) * number for k, number in t.items())
    invariants = {
        "number_of_lines": n,
        "rank": 3 if rank_three else "less_than_3",
        "t_vector": dict(sorted(t.items())),
        "r_vector": dict(sorted(r.items())),
        "maximum_multiplicity": max(t),
        "f_vector_projective": [f0, f1, f2],
        "melchior_defect": delta,
        "three_f2_minus_two_f1": 3 * f2 - 2 * f1,
        "simplicial": rank_three and delta == 0,
        "irreducible": rank_three and max(t) < n - 1,
        "characteristic_polynomial_descending_coefficients": [1, -n, b2, n - 1 - b2],
    }
    return invariants, points, counts


def verify_paper_invariants(invariants, lines, orbits, counts):
    """Verify the theorem's values and add its derived conclusions.

    Two classification facts used here are explained in the manuscript:
    reducible essential rank-three arrangements are near-pencils, and the
    two infinite irreducible families have sizes 2n and 4m+1. Supersolvable
    arrangements have characteristic polynomials splitting into integer
    linear factors. No classification of sporadic arrangements is assumed.
    """
    expected = {
        "number_of_lines": 51, "rank": 3,
        "t_vector": {2: 120, 3: 190, 5: 36, 6: 15},
        "r_vector": {20: 21, 22: 30}, "maximum_multiplicity": 6,
        "f_vector_projective": [361, 1080, 720], "melchior_defect": 0,
        "three_f2_minus_two_f1": 0, "simplicial": True, "irreducible": True,
        "characteristic_polynomial_descending_coefficients": [1, -51, 719, -669],
    }
    require(invariants == expected, "all computed invariants equal the paper's values")
    orbit_vertex_counts = [
        dict(sorted(Counter(counts[lines.index(line)] for line in orbit).items()))
        for orbit in orbits
    ]
    require(orbit_vertex_counts == [{20: 15}, {20: 6}, {22: 30}],
            "vertices per line in each orbit")
    n = invariants["number_of_lines"]
    polynomial = invariants["characteristic_polynomial_descending_coefficients"]
    b = polynomial[1] + 1
    c = polynomial[2] + b
    require([1, b - 1, c - b, -c] == polynomial,
            "characteristic polynomial factors as (T-1)(T^2+bT+c)")
    discriminant = b * b - 4 * c
    require((b, c, discriminant) == (-50, 669, -176),
            "quadratic factor and its negative discriminant")
    require(2 * invariants["f_vector_projective"][2] == 1440,
            "1440 chambers on the sphere")
    sporadic = invariants["irreducible"] and n % 2 == 1 and n % 4 == 3
    require(sporadic, "line count excludes both infinite irreducible families")
    invariants.update({
        "number_of_line_pairs": n * (n - 1) // 2,
        "number_of_line_vertex_dot_products": n * invariants["f_vector_projective"][0],
        "number_of_spherical_chambers": 2 * invariants["f_vector_projective"][2],
        "vertices_per_line_by_orbit": {
            name: histogram for name, histogram in zip(ORBIT_NAMES, orbit_vertex_counts)
        },
        "characteristic_polynomial_factors_descending_coefficients": [[1, -1], [1, b, c]],
        "quadratic_factor_discriminant": discriminant,
        "sporadic": sporadic,
        "supersolvable": False,
    })


def coefficient_strings(vector):
    return [[str(coefficient) for coefficient in a] for a in vector]


def certificate_document(lines, orbits, points, counts, invariants, point_orbits):
    """Return the deterministic certificate dictionary (integer keys internally).

    JSON serialization turns integer histogram keys into strings. Coordinates
    always use rational strings. Lines are grouped O15,O6,O30 and sorted within
    each orbit; vertices are globally sorted. Point-orbit order is Table 1.
    """
    orbit_tag = {line: name for name, orbit in zip(ORBIT_NAMES, orbits) for line in orbit}
    vertices = sorted(points)
    return {
        "schema_version": 1,
        "description": "Exact real projective A51 line arrangement and complete incidence certificate",
        "coordinate_convention": "[a0,a1,a2,a3] represents a0+a1*s+a2*s^2+a3*s^3; rational strings; one-based IDs",
        "number_field": {"poly": ["-1", "0", "-1", "0", "1"],
                         "interval": ["5/4", "13/10"], "real_generator": "sqrt((1+sqrt(5))/2)"},
        "projective_normalization": "the first nonzero coordinate is 1",
        "group": {"name": "icosahedral rotation group I", "order": 60,
                  "generator_names": list(GENERATOR_NAMES),
                  "generator_orders": list(GENERATOR_ORDERS)},
        "generators": [[coefficient_strings(row) for row in matrix] for matrix in GENERATORS],
        "seeds": [coefficient_strings(seed) for seed in SEEDS],
        "lines": [{"id": i + 1, "orbit": orbit_tag[line],
                   "normal": coefficient_strings(line), "number_of_vertices": counts[i]}
                  for i, line in enumerate(lines)],
        "vertices": [{"id": j + 1, "point": coefficient_strings(point),
                      "incident_lines": [i + 1 for i in sorted(points[point])],
                      "multiplicity": len(points[point])}
                     for j, point in enumerate(vertices)],
        "point_orbits": point_orbits,
        "invariants": invariants,
    }


def export_certificate(path, lines, orbits, points, counts, invariants, point_orbits):
    document = certificate_document(lines, orbits, points, counts, invariants, point_orbits)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def verify_construction():
    """Run the paper's complete exact check and return all computed data.

    The result dictionary contains group, lines, orbits, points, counts,
    invariants, point_orbits, and base_invariants. This function writes nothing
    and prints nothing, so companion scripts can reuse it.
    """
    verify_field()
    group = generate_group()
    lines, orbits = generate_lines(group)
    require(compact_line_orbits() == orbits,
            "independent compact formulas match all three rotation-group orbits")
    invariants, points, counts = analyze_lines(lines)
    point_orbits = verify_point_orbits(group, lines, orbits, points)
    verify_special_incidences(group, orbits)
    verify_paper_invariants(invariants, lines, orbits, counts)
    base, _, _ = analyze_lines(orbits[0] + orbits[1])
    require(base["t_vector"] == {2: 30, 3: 10, 4: 15, 5: 6}
            and base["simplicial"], "the 21-line base arrangement")
    return {
        "group": group, "lines": lines, "orbits": orbits, "points": points,
        "counts": counts, "invariants": invariants, "point_orbits": point_orbits,
        "base_invariants": base,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-json", type=Path, metavar="PATH",
                        help="write all 51 normals, 361 vertices, and incidences")
    parser.add_argument("--self-test", action="store_true",
                        help="delete a representative of each orbit and verify that simpliciality fails")
    args = parser.parse_args()
    start = time.monotonic()
    result = verify_construction()
    lines, orbits = result["lines"], result["orbits"]
    invariants, points, counts = result["invariants"], result["points"], result["counts"]
    point_orbits = result["point_orbits"]
    print("Exact A51 verification: PASS")
    print("Field: Q[s]/(s^4-s^2-1), with 5/4 < s < 13/10; phi=s^2")
    print("Rotation group: order 60; generator orders 3, 2, 5")
    print("Generators orthogonal, determinant +1; all rotations preserve the twelve vertices")
    print("Disjoint projective line orbits: 15 + 6 + 30 = 51")
    print("Independent compact coordinate formulas: PASS")
    print("All 1275 line pairs and all 18411 line/vertex incidences checked")
    print("All 12 displayed point orbits: sizes, incidence profiles, disjointness, completeness PASS")
    print("Both explicit special-incidence constructions at s^2=phi: PASS")
    print("Base arrangement A(21,2): t2=30, t3=10, t4=15, t5=6; simplicial")
    print("Rank: 3")
    print("t-vector: t2=120, t3=190, t5=36, t6=15; all other entries zero")
    print("r-vector: r20=21, r22=30; all other entries zero")
    print("Vertices per line by orbit: O15:20, O6:20, O30:22")
    print("Projective f-vector: (361, 1080, 720)")
    print("Spherical triangular chambers: 1440")
    print("Melchior defect: 120 - 3 - 2*36 - 3*15 = 0")
    print("Simplicial: 3*720 = 2*1080")
    print("Characteristic polynomial: t^3-51*t^2+719*t-669 = (t-1)*(t^2-50*t+669)")
    print("Not supersolvable: the quadratic factor has discriminant -176")
    print("Irreducible: maximum vertex multiplicity 6 is less than 50")
    print("Standard infinite families excluded: not a near-pencil; 51 is odd and 51 = 3 mod 4")
    print("Sporadic: excluded from near-pencils and the families R(1), R(2)")
    if args.self_test:
        for name, orbit in zip(ORBIT_NAMES, orbits):
            remaining = [line for line in lines if line != orbit[0]]
            deletion, _, _ = analyze_lines(remaining)
            require(not deletion["simplicial"] and deletion["melchior_defect"] > 0,
                    "deleting one line of " + name + " destroys simpliciality")
            print("Deletion self-test " + name + ": PASS; defect=" + str(deletion["melchior_defect"]))
    if args.export_json:
        export_certificate(args.export_json, lines, orbits, points, counts, invariants, point_orbits)
        print("Complete exact incidence certificate written to " + str(args.export_json))
    print("Elapsed seconds: {:.3f}".format(time.monotonic() - start))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ZeroDivisionError, KeyError, TypeError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
