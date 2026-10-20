#!/usr/bin/env python3
"""Verify the geometric mechanism in Section 4 of the A51 paper.

Run ``python3 verify_geometry.py`` beside ``verify_a51.py``. Python 3.9+
and its standard library are sufficient. The calculations are exact in
K = Q[s]/(s^4-s^2-1), with phi=s^2 and s the positive real root.

The concurrence equations are checked as identities in the polynomial
ring K[t], not merely by substituting the successful parameter t=s.
The program also checks the four pencil parameter sets, their projective
involution, and the cross-ratio that obstructs a realization of this
particular projective arrangement over Q(sqrt(5)).
"""

from verify_a51 import (
    ZERO, ONE, S, PHI, PHI_INV, HALF, GENERATORS, SEEDS,
    add, sub, neg, mul, inv, dot, normalize,
    matvec, matmul, transpose, generate_group, generate_lines,
    analyze_lines, verify_field, require,
)


# A polynomial is a tuple of field coefficients in ascending powers of t.
# Its representation is canonical: zero is (ZERO,), with no trailing zeros.
def poly(coefficients):
    coefficients = list(coefficients)
    while len(coefficients) > 1 and coefficients[-1] == ZERO:
        coefficients.pop()
    return tuple(coefficients) if coefficients else (ZERO,)


def pconstant(value):
    return (value,)


def padd(a, b):
    return poly(add(a[i] if i < len(a) else ZERO,
                    b[i] if i < len(b) else ZERO)
                for i in range(max(len(a), len(b))))


def pneg(a):
    return poly(neg(value) for value in a)


def psub(a, b):
    return padd(a, pneg(b))


def pmul(a, b):
    coefficients = [ZERO] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            coefficients[i + j] = add(coefficients[i + j], mul(x, y))
    return poly(coefficients)


def pscale(a, scalar):
    return poly(mul(value, scalar) for value in a)


def peval(a, value):
    result = ZERO
    for coefficient in reversed(a):
        result = add(mul(result, value), coefficient)
    return result


def pdot(a, b):
    result = (ZERO,)
    for x, y in zip(a, b):
        result = padd(result, pmul(x, y))
    return result


def pmatvec(matrix, vector):
    return tuple(pdot(tuple(pconstant(x) for x in row), vector)
                 for row in matrix)


def pcross(a, b):
    return tuple(psub(pmul(a[(i + 1) % 3], b[(i + 2) % 3]),
                      pmul(a[(i + 2) % 3], b[(i + 1) % 3]))
                 for i in range(3))


def pencil_parameters(lines):
    """Return finite field parameters t, with None denoting infinity."""
    return {
        None if line[1] == ZERO else mul(line[2], inv(line[1]))
        for line in lines if line[0] == ZERO
    }


def pencil_involution(line):
    """On [0:b:c], send t=c/b to phi/t, including 0 and infinity."""
    require(line[0] == ZERO, "the line belongs to the pencil through P")
    return normalize((ZERO, line[2], mul(PHI, line[1])))


def pencil_cross_ratio(a, b, c, d):
    """Convention CR(0,infinity;c,d)=c/d."""
    require(all(line[0] == ZERO for line in (a, b, c, d)),
            "the four lines belong to the pencil through P")

    def determinant(u, v):
        return sub(mul(u[1], v[2]), mul(u[2], v[1]))

    numerator = mul(determinant(a, c), determinant(b, d))
    denominator = mul(determinant(a, d), determinant(b, c))
    require(denominator != ZERO, "the cross-ratio denominator is nonzero")
    return mul(numerator, inv(denominator))


def in_quadratic_subfield(value):
    """Q(sqrt(5))=Q(phi) consists precisely of a0+a2*s^2."""
    return value[1] == 0 and value[3] == 0


def verify_geometry(group=None, line_orbits=None):
    """Run all geometric checks and return their computed finite data."""
    verify_field()
    if group is None:
        group = generate_group()
    if line_orbits is None:
        _, line_orbits = generate_lines(group)
    base = line_orbits[0] + line_orbits[1]
    point_p = (ONE, ZERO, ZERO)
    kappa, eta, _ = GENERATORS
    require(sum(dot(line, point_p) == ZERO for line in base) == 4,
            "P=(1:0:0) is a fourfold point of B")
    require(matvec(eta, point_p) == point_p, "eta fixes the axis point P")

    def half_matrix(rows):
        return tuple(tuple(mul(HALF, value) for value in row) for row in rows)

    g1 = half_matrix(((PHI_INV, ONE, PHI),
                      (neg(ONE), PHI, neg(PHI_INV)),
                      (neg(PHI), neg(PHI_INV), ONE)))
    g2 = half_matrix(((PHI_INV, neg(ONE), PHI),
                      (ONE, PHI, PHI_INV),
                      (neg(PHI), PHI_INV, ONE)))
    require(g1 in group and g2 in group, "the displayed g1 and g2 belong to I")

    zero = (ZERO,)
    one = (ONE,)
    t = (ZERO, ONE)
    ell_t = (zero, one, t)
    two = add(ONE, ONE)
    minus_two = neg(two)
    l1_plus = ((ONE, PHI), (PHI, neg(PHI_INV)), (neg(PHI_INV), ONE))
    l2_plus = ((neg(ONE), PHI), (PHI, PHI_INV), (PHI_INV, ONE))
    l1_minus = (pneg(l1_plus[0]), l1_plus[1], l1_plus[2])
    l2_minus = (pneg(l2_plus[0]), l2_plus[1], l2_plus[2])
    rotations = (g1, matmul(eta, g1), g2, matmul(eta, g2))
    four_polynomial_lines = (l1_plus, l1_minus, l2_plus, l2_minus)
    for rotation, factor, expected in zip(
            rotations, (two, minus_two, two, minus_two), four_polynomial_lines):
        actual = tuple(pscale(coordinate, factor)
                       for coordinate in pmatvec(rotation, ell_t))
        require(actual == expected, "the four displayed normals agree in K[t]")

    q1 = (zero, (neg(PHI_INV), ONE), (neg(PHI), PHI_INV))
    q2 = (zero, (PHI_INV, ONE), (neg(PHI), neg(PHI_INV)))
    for point, pair in ((q1, (l1_plus, l1_minus)),
                        (q2, (l2_plus, l2_minus))):
        require(all(pdot(line, point) == zero for line in pair),
                "each Q_i(t) lies on its two displayed lines identically")
    coincidence = psub(pmul(q1[1], q2[2]), pmul(q1[2], q2[1]))
    expected_coincidence = (two, ZERO, neg(mul(two, PHI_INV)))
    require(coincidence == expected_coincidence,
            "the coincidence determinant is exactly 2-2*t^2/phi")
    require(coincidence == pscale(psub(pmul(t, t), pconstant(PHI)),
                                 neg(mul(two, PHI_INV))),
            "the fivefold concurrence equation is equivalent to t^2=phi")
    for root in (S, neg(S)):
        point = normalize((ZERO, ONE, neg(root)))
        require(all(normalize(tuple(peval(x, root) for x in q)) == point
                    for q in (q1, q2)),
                "Q1 and Q2 coincide at (0:1:-t) when t=+/-s")
        four_lines = {
            normalize(tuple(peval(x, root) for x in line))
            for line in four_polynomial_lines
        }
        require(len(four_lines) == 4 and four_lines.issubset(set(line_orbits[2])),
                "four distinct added lines occur at each of t=+/-s")
        require(all(dot(line, point) == ZERO for line in four_lines),
                "the four added lines are incident with the fivefold point")
        mirror = normalize((ONE, ZERO, ZERO))
        require(mirror not in four_lines and dot(mirror, point) == ZERO,
                "the mirror is a fifth, distinct concurrent line")

    second_t = pmatvec(matmul(eta, transpose(kappa)), ell_t)
    require(second_t == (one, pneg(t), zero),
            "eta*kappa^(-1) sends ell_t to [1:-t:0]")
    triple_t = (pmul(t, t), t, pconstant(neg(ONE)))
    require(pcross(ell_t, second_t) == triple_t,
            "the two added lines meet at (t^2:t:-1)")
    old_line = matvec(kappa, SEEDS[1])
    require(old_line == (ONE, ZERO, PHI),
            "the displayed polar line is kappa*[0:phi:1]")
    triple_condition = pdot(tuple(pconstant(x) for x in old_line), triple_t)
    require(triple_condition == (neg(PHI), ZERO, ONE),
            "the polar-line concurrence equation is exactly t^2-phi")

    o10 = sorted({normalize(matvec(g, (ZERO, PHI_INV, PHI))) for g in group})
    require(len(o10) == 10 and not set(o10).intersection(base),
            "O10 has ten projective lines and is disjoint from B")
    phi_squared = mul(PHI, PHI)
    expected_parameters = (
        {ZERO, None}, {PHI_INV, neg(PHI_INV)},
        {phi_squared, neg(phi_squared)}, {S, neg(S)},
    )
    for orbit, expected in zip(
            (line_orbits[0], line_orbits[1], o10, line_orbits[2]),
            expected_parameters):
        require(pencil_parameters(orbit) == expected,
                "the displayed pencil parameter set is exact")

    mirror_zero = (ZERO, ONE, ZERO)
    mirror_infinity = (ZERO, ZERO, ONE)
    require(pencil_involution(mirror_zero) == normalize(mirror_infinity)
            and pencil_involution(mirror_infinity) == normalize(mirror_zero),
            "the involution exchanges the two mirrors")
    for sign in (ONE, neg(ONE)):
        polar = (ZERO, ONE, mul(sign, PHI_INV))
        threefold = (ZERO, ONE, mul(sign, phi_squared))
        fixed = (ZERO, ONE, mul(sign, S))
        require(pencil_involution(polar) == normalize(threefold)
                and pencil_involution(threefold) == normalize(polar),
                "the involution exchanges O6 and O10 lines of the same sign")
        require(pencil_involution(fixed) == normalize(fixed),
                "the added pencil line is fixed by the involution")
    # Fixed finite parameters satisfy t^2=phi; neither 0 nor infinity is fixed.
    fixed_condition = psub(pmul(t, t), pconstant(PHI))
    require(fixed_condition == triple_condition,
            "the involution fixed-line equation is the concurrence equation")

    cross_ratio_lines = (
        mirror_zero, mirror_infinity, (ZERO, ONE, S), (ZERO, ONE, PHI_INV),
    )
    all_lines = set().union(*(set(orbit) for orbit in line_orbits))
    require(all(normalize(line) in all_lines for line in cross_ratio_lines),
            "all four cross-ratio lines belong to A51")
    require(len({normalize(line) for line in cross_ratio_lines}) == 4,
            "the four cross-ratio lines are distinct")
    cross_ratio = pencil_cross_ratio(*cross_ratio_lines)
    s_cubed = mul(PHI, S)
    require(cross_ratio == s_cubed, "the cross-ratio is phi*s=s^3")
    require(in_quadratic_subfield(PHI) and not in_quadratic_subfield(cross_ratio),
            "the cross-ratio lies outside Q(phi)=Q(sqrt(5))")
    require(mul(PHI, sub(ONE, PHI)) == neg(ONE),
            "the norm of phi from Q(sqrt(5)) to Q is -1")

    base31, _, _ = analyze_lines(base + o10)
    require(base31["number_of_lines"] == 31
            and base31["t_vector"] == {2: 60, 3: 40, 5: 6, 6: 15}
            and base31["simplicial"],
            "B union O10 is a simplicial 31-line arrangement")
    return {
        "mirror_concurrence_polynomial": coincidence,
        "polar_concurrence_polynomial": triple_condition,
        "cross_ratio": cross_ratio,
        "pencil_parameters": expected_parameters,
        "base31_t_vector": base31["t_vector"],
    }


def main():
    verify_geometry()
    print("Exact Section 4 geometry verification: PASS")
    print("Fourfold base point P and displayed rotations g1, g2: PASS")
    print("All four moving-line formulas and both Q_i(t): identities checked in K[t]")
    print("Mirror concurrence: 2 - 2*t^2/phi = 0")
    print("Polar-line concurrence: t^2 - phi = 0")
    print("At t=+/-s, four distinct added lines concur with a mirror")
    print("Pencil parameters: O15={0,infinity}; O6={+/-phi^(-1)};")
    print("                   O10={+/-phi^2}; O30={+/-s}")
    print("Involution t -> phi/t: exchanges the stated old lines; fixes exactly +/-s")
    print("Cross-ratio CR(0,infinity;s,phi^(-1)) = s^3, outside Q(sqrt(5))")
    print("B union O10: 31 lines; t2=60, t3=40, t5=6, t6=15; simplicial")


if __name__ == "__main__":
    main()
