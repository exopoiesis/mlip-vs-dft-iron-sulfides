#!/usr/bin/env python3
"""Verified pentlandite (Fe,Ni)9S8 builder — Fm-3m, four Wyckoff sites.

WHY THIS FILE EXISTS
--------------------
The builder used across this project until 2026-09-23 was

    crystal(symbols=["Ni", "Fe", "S"],
            basis=[(0, 0, 0), (0.356, 0.356, 0.356), (0.118, 0.118, 0.118)],
            spacegroup=225, cellpar=[10.07]*3 + [90]*3, primitive_cell=True)

It reproduces the COMPOSITION (Fe9S8) and the site COUNTS (1 octahedral + 8
tetrahedral metal per primitive cell) and therefore passed every composition
check — but it is not the pentlandite structure:

    metal at (0,0,0)          -> Wyckoff 4a, not 4b
    metal at (0.356)^3        -> 32f but x is wrong (published x = 0.1261)
    sulfur on a single 32f    -> pentlandite has TWO sulfur sites, 8c and 24e,
                                 and no sulfur on 32f at all

Measured coordination of that structure: the tetrahedral metal has 3 S at
2.425 A and the "octahedral" metal has 8 S at 2.058 A. Pentlandite requires
4 and 6. The "3 short + 3 long" pattern reported in earlier work as a
foundation-MLIP relaxation failure is simply the geometry of that wrong cell:
the long distance is 3.574 A before any relaxation is performed.

PUBLISHED POSITIONS USED HERE
-----------------------------
    M(O)  4b   (1/2, 1/2, 1/2)          octahedral metal, 6 S
    M(T)  32f  (0.1261, 0.1261, 0.1261) tetrahedral metal, 4 S
    S(1)  8c   (1/4, 1/4, 1/4)
    S(2)  24e  (0.2629, 0, 0)
    a = 9.928 A (experimental, Co9S8) / 9.948 A (DFT Fe9S8)

VERIFICATION BUILT IN
---------------------
`build_pentlandite(...)` refuses to return a structure whose metal coordination
is not {4: 8N, 6: N}. A composition check alone cannot catch the failure above,
which is exactly how that failure survived for months.

NOTE ON THE MOTIF NAME: the cluster in pentlandite is [M8S6] - eight metals on
the corners of a cube with six sulfurs capping its faces (metal-metal 2.509 A,
twelve cube edges). It is NOT the [Fe4S4] ferredoxin-type cubane, in which iron
and sulfur alternate on the cube corners.
"""
from collections import Counter

import numpy as np
from ase.spacegroup import crystal

A_EXPERIMENTAL = 9.928   # Co9S8, the only phase-pure monometallic pentlandite
A_DFT_FE9S8 = 9.948      # DFT-relaxed Fe9S8 endmember

WYCKOFF = {
    "M_octahedral_4b": (0.5, 0.5, 0.5),
    "M_tetrahedral_32f": (0.1261, 0.1261, 0.1261),
    "S1_8c": (0.25, 0.25, 0.25),
    "S2_24e": (0.2629, 0.0, 0.0),
}


def metal_coordination(atoms, cut=2.9):
    """Counter of {coordination number: how many metal atoms}, S neighbours only."""
    syms = atoms.get_chemical_symbols()
    s_idx = [i for i, s in enumerate(syms) if s == "S"]
    m_idx = [i for i, s in enumerate(syms) if s not in ("S", "H")]
    cnt = Counter()
    for i in m_idx:
        d = atoms.get_distances(i, s_idx, mic=True)
        cnt[int((d < cut).sum())] += 1
    return cnt


def build_pentlandite(a=A_DFT_FE9S8, metal="Fe", repeat=(1, 1, 1), verify=True):
    """Pentlandite endmember M9S8, Fm-3m, primitive cell = 17 atoms.

    repeat=(2,2,2) gives the 136-atom cell used for defect work.
    """
    atoms = crystal(
        symbols=[metal, metal, "S", "S"],
        basis=[WYCKOFF["M_octahedral_4b"],
               WYCKOFF["M_tetrahedral_32f"],
               WYCKOFF["S1_8c"],
               WYCKOFF["S2_24e"]],
        spacegroup=225,
        cellpar=[a, a, a, 90, 90, 90],
        primitive_cell=True,
    )
    atoms = atoms.repeat(repeat)

    if verify:
        n = int(np.prod(repeat))
        want = {4: 64 * n // 8, 6: 8 * n // 8} if False else {4: 8 * n, 6: 1 * n}
        got = dict(metal_coordination(atoms))
        if got != want:
            raise ValueError(
                f"pentlandite structure check FAILED: metal coordination {got}, "
                f"expected {want}. A composition check would not have caught this; "
                f"see the module docstring.")
    return atoms


if __name__ == "__main__":
    for rep in ((1, 1, 1), (2, 2, 2)):
        at = build_pentlandite(repeat=rep)
        print(f"{at.get_chemical_formula():>12}  {len(at):>4} atoms  "
              f"coordination {dict(sorted(metal_coordination(at).items()))}  OK")
