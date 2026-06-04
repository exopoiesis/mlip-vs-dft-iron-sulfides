#!/usr/bin/env python3
"""
Canonical V_Fe (octahedral 16d) + H lateral hop NEB -- greigite Fe3S4
(Fd-3m #227 setting=2), conventional cell = 56 atoms (Z=8, Fe24 S32);
after V_Fe + H = 56 atoms = Fe23 S32 H1.

Adapted from neb_canonical_greigite_56at_qe.py (V_S+H, deprecated s148) per
РЕШЕНИЕ-082 chemistry-signature extension (2026-05-20, s148):
  - V_S+H broken for ALL Fe-S sulfides где S all-equivalent under SG symmetry.
  - Greigite Fd-3m S 32e single orbit → V_S+H artifact predicted (mack/pent/marc precedent).
  - V_Fe pivot per Liu 2021 ACS Omega 0.26 eV anchor (extrapolated).

Greigite specifics retained from V_S+H parent (КРИТИЧНО, не упрощать):
  - nspin=2 + Hubbard U_eff=1.0 eV (Dudarev) ON Fe1+Fe2 sublattices [НЕОБХОДИМО]
  - Ferrimagnetic A↑↓B init: Fe_tet 8a (+5 μB Fe³⁺) ↑ / Fe_oct 16d (-4 μB) ↓
  - apply_greigite_ferri_split mandatory (sublattice labels Fe1/Fe2)
  - РЕШЕНИЕ-094 AFM+U Tier 1 recipe: plain mixing β=0.05, mixing_ndim=12,
    mixing_fixed_ns=15, tot_mag=0, conv_thr=1e-3, david diag
  - HUBBARD card injection POST-WRITER (QE 7.2+ syntax, ortho-atomic)

V_Fe choice: octahedral Fe (Wyckoff 16d). Pyrrhotite T11 (s135) + pentlandite
(s132) precedent — octahedral V_Fe для thiospinel/sulfide families. Tetrahedral
Fe_tet (8a, 4 S neighbours) NOT picked (would be "smaller pocket, similar to
mack" — но greigite has BOTH coordinations available — clear choice for clean
6-S V_Fe lateral hop).

Reaction coordinate (this script):
  - Build greigite Fd-3m setting=2 conv cell (56 atoms pristine Fe24 S32).
  - Apply ferri split via apply_greigite_ferri_split → Fe1=tet, Fe2=oct.
  - Pick central Fe_oct atom (V_Fe = vacancy_atom) — Fe2 only, octahedral 6-S.
  - Find 6 S neighbours within fe_s_max ~2.85 Å (greigite Fe_oct-S ~2.46 nominal).
  - Pick 2 S atoms (S_i, S_k) с maximum d(S_i, S_k) — lateral hop через V_Fe pocket.
  - Build endA: H placed 1.35 Å от S_i toward V_Fe (S-H в pocket).
  - Build endB: H placed 1.35 Å от S_k toward V_Fe.
  - PRE-FLIGHT GATES (s148, mandatory): G1 Wyckoff, G2 parity, G3 pocket-radius.
  - Relax pristine (REUSE skip if --reuse-pristine) + endA + endB BFGS.
  - CI-NEB FIRE 9 images.

Expected E_a: 0.2-0.5 eV bulk Fe-S Fe_oct V_Fe (Liu 2021 surface anchor 0.26 eV).
Может быть выше mack/pent due AFM+U coupling complications + larger cell relaxation.

==========================================================================
ORIGINAL V_S+H notes (DEPRECATED but retained для reference):
==========================================================================


QE PWSCF (planewave, no Pulay forces). Greigite is FERRIMAGNETIC inverse-thiospinel
(A↑↓B sublattices): Fe_tet (8a, +5 μB Fe³⁺ HS d⁵) ↑ vs Fe_oct (16d, -4 μB
Fe²⁺/Fe³⁺ mixed) ↓. Net moment ~3.5 μB / Fe3S4 f.u. (exp Spender 1972; Coey
1970). nspin=2 + Hubbard U_eff=1.0 eV (Dudarev) PBE is the physically correct
basis — diamagnetic single-spin treatment WOULD COLLAPSE the magnetic order
(verified s142: only nspin=2 + ferri init gives stable A↑↓B with m_Fe_tet≈+5,
m_Fe_oct≈-4 μB per atom; Kwon-Subedi 2011 PRB).

Hubbard U_eff=1.0 eV consensus (Dudarev simplified):
  - Devey 2009 J Phys Chem C  (U=1.0 eV)
  - Roldan & de Leeuw 2016 RSPA  (U=1.0 eV, CI-NEB H2O greigite surfaces)
  - 4 более recent papers triangulate same value (Q-115 NOMAD benchmarks).
  - applied to Fe1 (tet) AND Fe2 (oct) 3d shells, ortho-atomic projection.

Canonical protocol (apples-to-apples with mack/pent/marc V_S+H siblings):
  - V_S at one S site, H hops S_i -> S_k via V_S pocket
  - Build via s142 build_greigite_conventional (a=9.876 Å, u=0.253)
  - Ferri init via s142 apply_greigite_ferri_split (fractional [-1,+1])
  - HUBBARD card injected POST-WRITER на Fe1-3d + Fe2-3d (ortho-atomic)
  - Full relax pristine + endpoints (NO FixAtoms)
  - CI-NEB 9 images, IDPP interpolation (с prewrap ASE #1130), FIRE optimizer

Output JSON includes magnetic_moments_endA/B + fe_neighbor_type_endA (tet/oct/mixed)
для H-anchor site classification (does H prefer S near Fe_tet vs Fe_oct?).

Workflow precedent: Roldan & de Leeuw 2016 RSPA "First-principles theoretical
study of greigite (Fe3S4) surfaces and their interaction with H2O" — same
SG, same U=1.0, plain mixing, atomic_random startingwfc, conv_thr ~1e-6.

Per РЕШЕНИЕ-079 (s125) + РЕШЕНИЕ-094 (s142 AFM+U recipe greigite):
  - QE PWSCF (no Pulay forces, paper-grade fmax achievable)
  - nspin=2 + Hubbard U=1.0 (ferri ground state mandatory; Tier 2 production)
  - plain mixing β=0.05 ndim=12 fixed_ns=15  (s142 RESHENIE-094 transferable recipe)
  - david diagonalization, diago_thr_init=1e-4, diago_full_acc=True
  - startingwfc='atomic+random' (Devey 2009 / Roldan 2016 convention)
  - conv_thr=1e-6 Ry (Tier 2 production; smearing degauss=0.01 Ry s142-recipe per R1-FIX F1)
  - BFGS endpoints (PWFFT clean forces), FIRE NEB (apples-to-apples)

References:
  - knowledge/Q115_NOMAD_BENCHMARKS.md (U=1.0 consensus, 6 papers)
  - knowledge/handoffs/SESSION_HANDOFF_2026-05-10_s142.md (РЕШЕНИЕ-094)
  - experiments/2026-05-19_greigite_neb/docs/LITERATURE_FINDINGS.md
  - Roldan & de Leeuw 2016 RSPA (CI-NEB H2O on greigite surfaces precedent)
  - Devey, Roldan, de Leeuw 2009 J Phys Chem C (Hubbard U calibration)
  - infra/gpu_scripts/afmu_singlepoint_greigite.py (s142 building blocks reused)
  - infra/gpu_scripts/neb_batch_qe_v1_ntyp_split_patch.py (label normaliser)
  - infra/gpu_scripts/neb_canonical_marc_96at_qe.py (parent template, nspin=1 sibling)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from ase import Atom
from ase.spacegroup import crystal
from ase.optimize import BFGS, LBFGSLineSearch, FIRE
from ase.mep import DyNEB, NEB
from ase.io import read, write
from ase.geometry import find_mic
from ase.calculators.espresso import Espresso, EspressoProfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_neb_band(images, args):
    """Create production NEB band using the selected optimizer policy."""
    if args.dyneb:
        neb = DyNEB(
            images,
            climb=True,
            method="improvedtangent",
            k=args.k_spring,
            fmax=args.fmax_neb,
            dynamic_relaxation=True,
            scale_fmax=args.dyneb_scale_fmax,
        )
        policy = "DyNEB"
    else:
        neb = NEB(images, climb=True, method="improvedtangent", k=args.k_spring)
        policy = "NEB"
    print(
        f"[NEB POLICY] {policy}: dynamic_relaxation={bool(args.dyneb)}, "
        f"scale_fmax={args.dyneb_scale_fmax}, fmax={args.fmax_neb}",
        flush=True,
    )
    return neb, {
        "neb_policy": policy,
        "dyneb_enabled": bool(args.dyneb),
        "dyneb_dynamic_relaxation": bool(args.dyneb),
        "dyneb_scale_fmax": float(args.dyneb_scale_fmax),
    }

# GREIGITE NEW: import s142 building blocks + label-split normaliser.
# Search both deployment locations (workspace on Vast.ai, local infra/gpu_scripts).
import sys as _sys
from pathlib import Path as _Path
for _candidate in (
    _Path("/workspace/infra/gpu_scripts"),
    _Path("/workspace"),
    _Path(__file__).resolve().parent.parent.parent.parent / "infra" / "gpu_scripts",
):
    if _candidate.exists() and str(_candidate) not in _sys.path:
        _sys.path.insert(0, str(_candidate))
try:
    from afmu_singlepoint_greigite import (
        build_greigite_conventional,
        apply_greigite_ferri_split,
        identify_fe_sites_by_coordination,
    )
except ImportError as _imp_err:
    print(f"[WARN] afmu_singlepoint_greigite import failed: {_imp_err}", flush=True)
    build_greigite_conventional = None
    apply_greigite_ferri_split = None
    identify_fe_sites_by_coordination = None
try:
    from neb_batch_qe_v1_ntyp_split_patch import normalise_split_labels_in_pwi
except ImportError as _imp_err:
    print(f"[WARN] normalise_split_labels_in_pwi import failed: {_imp_err}", flush=True)
    normalise_split_labels_in_pwi = None


# GREIGITE FIX: rename mineral tag + per-mineral lockfile
MINERAL_TAG = "greigite_56at_qe_VFe"
LOCK_FILE = Path("/workspace/.neb_canonical_greigite_qe_VFe.lock")

PW_BIN = os.environ.get("PW_BIN", "pw.x")              # resolve via PATH inside image
# ONCV-SR PBE (Pseudo Dojo nc-sr-04 standard) per Q115 protocol §1.
# W3 default after `bash /opt/pp/download_pp.sh` (PSL PAW) — secondary;
# ONCV downloaded by `tmp/go_w3_download_oncv.sh` to /opt/pp/oncv_pbe.
PSEUDO_DIR = os.environ.get("ESPRESSO_PSEUDO",
                            os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def acquire_singleton():
    """python3-only singleton (rule #31, ps -C filter, not pgrep -f)."""
    try:
        out = subprocess.check_output(
            ["ps", "-C", "python3", "-o", "pid=,args="], text=True
        )
        my_pid = os.getpid()
        for line in out.strip().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue
            pid_str, args_str = parts
            # GREIGITE FIX: singleton scoped к greigite script name
            if "neb_canonical_greigite_56at_qe_VFe.py" not in args_str:
                continue
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid == my_pid:
                continue
            print(f"[SINGLETON] another instance pid={pid}: {args_str}", flush=True)
            sys.exit(2)
    except subprocess.CalledProcessError:
        pass
    if LOCK_FILE.exists():
        try:
            old = int(LOCK_FILE.read_text().strip())
            os.kill(old, 0)
            print(f"[SINGLETON] lock held by pid={old}, exit", flush=True)
            sys.exit(2)
        except (OSError, ValueError):
            print("[SINGLETON] stale lock, overwriting", flush=True)
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))


def release_singleton():
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception:
        pass


# ============================================================
# Builder -- canonical greigite Fd-3m setting=2 conventional cell = 56 atoms
# (Fe³⁺ at 8a tet, Fe²⁺/³⁺ mixed at 16d oct, S at 32e u≈0.253)
# ============================================================

def build_greigite_56at(a=9.876, u=0.253, repeat=(1, 1, 1)):
    """Greigite Fe3S4, Fd-3m setting=2, a=9.876 Å, conventional cell = 56 atoms.

    GREIGITE NEW: thin wrapper around s142 build_greigite_conventional()
    (afmu_singlepoint_greigite.py). Falls back to inline crystal() call с
    same Wyckoff data if s142 module unavailable.

    Atom order (Wyckoff preserved by ASE crystal()):
        indices [0:8]   — Fe_tet (8a, magmom +1.0 ferri init → +5 μB Fe³⁺ HS)
        indices [8:24]  — Fe_oct (16d, magmom −0.9 ferri init → −4 μB Fe²⁺/³⁺ mixed)
        indices [24:56] — S (32e u≈0.253)

    Cell ~9.88 Å — already > 8 Å V_S–V_S spacing (no supercell needed для
    one-vacancy NEB; --repeat (1,1,1) default).
    Phase 2 option: repeat=(2,1,1) → 112 atoms finite-size convergence check.
    """
    if build_greigite_conventional is not None:
        unit = build_greigite_conventional(a_angstrom=a, u_param=u)
    else:
        # GREIGITE NEW: inline fallback (mirror s142 logic verbatim)
        unit = crystal(
            symbols=["Fe", "Fe", "S"],
            basis=[
                (0.125, 0.125, 0.125),    # 8a tet
                (0.500, 0.500, 0.500),    # 16d oct
                (u, u, u),                 # 32e
            ],
            spacegroup=227,
            setting=2,                     # ITA standard origin choice 2
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=False,          # MUST be False (primitive = FCC 14 at)
        )
        n_fe = sum(1 for s in unit.get_chemical_symbols() if s == "Fe")
        n_s = sum(1 for s in unit.get_chemical_symbols() if s == "S")
        assert n_fe == 24 and n_s == 32 and len(unit) == 56, (
            f"Greigite build wrong multiplicity: Fe={n_fe} S={n_s} N={len(unit)}"
        )
    return unit.repeat(tuple(repeat))


# ============================================================
# V_Fe (octahedral 16d) picker — s148 V_Fe pivot for greigite
# ============================================================

def pick_vfe_oct_and_two_s_neighbours(atoms, oct_indices, fe_s_max=2.85,
                                       expected_n_s=6):
    """Pick V_Fe = Fe_oct nearest cell centre + 2 S neighbours (max d_S-S pair).

    Greigite has 16 Fe_oct (Wyckoff 16d) per conventional cell. Octahedral
    coordination = 6 S neighbours. fe_s_max=2.85 (Fe_oct-S nominal 2.46 Å).

    From 6 S, pick (S_i, S_k) с maximum d(S_i, S_k) within pocket = lateral hop
    через V_Fe pocket center.

    Returns: (fe_v_idx, s_i_idx, s_k_idx, hop_d, s_neighbours_list)
    """
    if not oct_indices:
        raise RuntimeError("No Fe_oct indices passed (apply_greigite_ferri_split first)")

    syms = atoms.get_chemical_symbols()
    # Validate all oct_indices are Fe
    for j in oct_indices:
        if not syms[j].startswith("Fe"):
            raise RuntimeError(f"oct_index {j} is not Fe: {syms[j]}")

    # Pick Fe_oct nearest cell centre (deterministic)
    cell_centre = atoms.cell.array.sum(axis=0) / 2
    fe_d_centre = [(float(np.linalg.norm(atoms.positions[i] - cell_centre)), i)
                   for i in oct_indices]
    fe_d_centre.sort()
    fe_v = fe_d_centre[0][1]

    # Find 6 S neighbours within fe_s_max
    s_indices = [i for i, s in enumerate(syms) if s == "S"]
    s_with_d = []
    for s_idx in s_indices:
        d = float(atoms.get_distance(fe_v, s_idx, mic=True))
        if d <= fe_s_max:
            s_with_d.append((d, s_idx))
    s_with_d.sort()
    if len(s_with_d) < expected_n_s:
        raise RuntimeError(
            f"V_Fe_oct at idx {fe_v} has only {len(s_with_d)} S within {fe_s_max} Å; "
            f"expected ≥{expected_n_s} for octahedral coordination."
        )
    s_neighbours = [s_idx for _, s_idx in s_with_d[:expected_n_s]]

    # Pair (S_i, S_k) max d(S_i, S_k) within pocket
    best_pair = None
    best_d = -1.0
    candidates = []
    n_neigh = len(s_neighbours)
    for a in range(n_neigh):
        for b in range(a + 1, n_neigh):
            si, sk = s_neighbours[a], s_neighbours[b]
            d_si_sk = float(atoms.get_distance(si, sk, mic=True))
            candidates.append((si, sk, d_si_sk))
            if d_si_sk > best_d:
                best_d = d_si_sk
                best_pair = (si, sk, d_si_sk)

    s_i, s_k, hop_d = best_pair

    print(f"[V_Fe_oct pick mineral=greigite] V_Fe_oct={fe_v}", flush=True)
    print(f"  6 S neighbours within {fe_s_max} Å (octahedral): {s_neighbours}", flush=True)
    for dist, s_idx in s_with_d[:expected_n_s]:
        print(f"    S idx {s_idx}: d_FeS={dist:.3f} Å", flush=True)
    print(f"  S-S pair candidates (descending d_S-S):", flush=True)
    for si_c, sk_c, d_c in sorted(candidates, key=lambda c: -c[2]):
        tag = " [SELECTED]" if (si_c, sk_c) == (s_i, s_k) else ""
        print(f"    S {si_c} <-> S {sk_c}: d={d_c:.3f} Å{tag}", flush=True)

    return fe_v, s_i, s_k, hop_d, s_neighbours


# ============================================================
# Triple picker + H placer (VERBATIM — DEPRECATED V_S+H, kept for ref)
# ============================================================

def pick_vacancy_and_two_neighbours(atoms, d_min=2.5, d_max=4.5,
                                     z_tolerance=None, mineral_name="",
                                     hop_mode="diagonal"):
    s_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "S"]

    def _within_layer(a, b):
        if z_tolerance is None:
            return True
        cz = atoms.cell.lengths()[2]
        dz = abs(atoms.positions[a, 2] - atoms.positions[b, 2])
        dz = min(dz, cz - dz)
        return dz <= z_tolerance

    def _search(lo, hi):
        candidates = []
        for sv in s_indices:
            pos_sv = atoms.positions[sv]
            nbrs = []
            for si in s_indices:
                if si == sv:
                    continue
                d_sv_si = atoms.get_distance(sv, si, mic=True)
                if lo < d_sv_si < hi and _within_layer(sv, si):
                    nbrs.append((si, d_sv_si))
            if len(nbrs) < 2:
                continue
            for ia in range(len(nbrs)):
                for ib in range(ia + 1, len(nbrs)):
                    si, _ = nbrs[ia]
                    sk, _ = nbrs[ib]
                    d_si_sk = atoms.get_distance(si, sk, mic=True)
                    if d_si_sk < 2.3:   # exclude S2 dimer (marcasite/pyrite; harmless for greigite — no S2 dimers, S-S min ~3.5 Å)
                        continue
                    if not _within_layer(si, sk):
                        continue
                    pos_si = atoms.positions[si]
                    pos_sk = atoms.positions[sk]
                    d_sk_from_si = pos_sk - pos_si
                    cell = atoms.cell.array
                    fc = np.linalg.solve(cell.T, d_sk_from_si)
                    fc -= np.round(fc)
                    d_sk_mic = cell.T @ fc
                    midpoint = pos_si + 0.5 * d_sk_mic
                    d_mid_sv_vec = pos_sv - midpoint
                    fc2 = np.linalg.solve(cell.T, d_mid_sv_vec)
                    fc2 -= np.round(fc2)
                    d_mid_sv_mic = cell.T @ fc2
                    d_mid_sv = float(np.linalg.norm(d_mid_sv_mic))
                    if hop_mode != "nearest" and d_mid_sv > 3.0:
                        continue
                    candidates.append({
                        "sv": sv, "si": si, "sk": sk,
                        "d_si_sk": float(d_si_sk),
                        "d_mid_sv": float(d_mid_sv),
                    })
        if not candidates:
            return None, []
        if hop_mode == "antipodal":
            candidates.sort(key=lambda c: (c["d_mid_sv"], c["d_si_sk"]))
        else:
            candidates.sort(key=lambda c: (c["d_si_sk"], c["d_mid_sv"]))
        return candidates[0], candidates

    best, candidates = _search(d_min, d_max)
    if best is None:
        best, candidates = _search(1.5, 5.5)
    if best is None:
        raise RuntimeError("No valid (V_S, S_i, S_k) triple found")

    print(f"[hop_diag mineral={mineral_name} mode={hop_mode}] top-5 triples "
          f"(by d_si_sk, d_mid_sv):", flush=True)
    for rank, c in enumerate(candidates[:5], start=1):
        tag = " [SELECTED]" if rank == 1 else ""
        print(f"  {rank}. sv={c['sv']} si={c['si']} sk={c['sk']} "
              f"d_si_sk={c['d_si_sk']:.3f} d_mid_sv={c['d_mid_sv']:.3f}{tag}",
              flush=True)
    return best["sv"], best["si"], best["sk"], best["d_si_sk"]


def prewrap_endpoint_for_idpp(initial, final, label="endB"):
    """
    Fix ASE GitLab issue #1130: pre-IDPP linear interpolation does NOT apply
    minimum-image convention, so atoms whose endpoint position wraps across
    PBC get linearly interpolated через WHOLE cell (not the short hop). This
    creates broken initial NEB path with atom overlaps.

    Fix: pre-wrap final endpoint relative to initial via `find_mic`, so
    `final.positions = initial.positions + mic_displacement`. After this,
    naive linear interp `initial + i*disp` gives correct path.

    Empirically verified s128 (pyr 96at): only H atom wraps across z-boundary
    (Δz_naive=10.0 Å vs Δz_mic=-0.84 Å). Heavy atoms unaffected.

    Returns: (final_unwrapped, n_unwrapped, summary) — modified Atoms +
    count + dict с forensic info (max |Δ_mic|, atom symbols).
    """
    # Physicist condition #1: defensive guard against slab/2D systems where
    # find_mic для non-PBC axis silently returns naive displacement.
    assert all(initial.pbc), (
        f"prewrap_endpoint_for_idpp requires full 3D PBC, got pbc={initial.pbc}. "
        f"For slabs, manual unwrap needed for non-PBC axis."
    )
    disp_naive = final.positions - initial.positions
    disp_mic, _ = find_mic(disp_naive, initial.cell, initial.pbc)
    diffs = np.linalg.norm(disp_naive - disp_mic, axis=1)
    wrapped_idx = np.where(diffs > 0.5)[0]
    summary = {
        "n_unwrapped":   int(len(wrapped_idx)),
        "max_disp_naive_A": float(np.linalg.norm(disp_naive, axis=1).max()),
        "max_disp_mic_A":   float(np.linalg.norm(disp_mic, axis=1).max()),
        "wrapped_atoms": [
            {"idx": int(i), "symbol": final.get_chemical_symbols()[i],
             "naive_A": float(np.linalg.norm(disp_naive[i])),
             "mic_A":   float(np.linalg.norm(disp_mic[i]))}
            for i in wrapped_idx
        ],
    }
    if len(wrapped_idx) == 0:
        return final, 0, summary
    out = final.copy()
    out.positions = initial.positions + disp_mic
    print(f"[IDPP-PREWRAP {label}] {len(wrapped_idx)} atom(s) unwrapped: "
          f"{[(int(i), final.get_chemical_symbols()[i]) for i in wrapped_idx]}",
          flush=True)
    for i in wrapped_idx:
        sym = final.get_chemical_symbols()[i]
        print(f"  atom {i} ({sym}): naive |Δ|={np.linalg.norm(disp_naive[i]):.3f} Å "
              f"→ mic |Δ|={np.linalg.norm(disp_mic[i]):.3f} Å", flush=True)
    return out, len(wrapped_idx), summary


def place_h_on_neighbour_s(atoms, v_idx, h_anchor_idx, bond_length=1.35):
    pos_v = atoms.positions[v_idx].copy()
    pos_a = atoms.positions[h_anchor_idx].copy()
    dvec = pos_v - pos_a
    cell = atoms.cell.array
    fc = np.linalg.solve(cell.T, dvec)
    fc -= np.round(fc)
    dvec_mic = cell.T @ fc
    dn = np.linalg.norm(dvec_mic)
    if dn < 1e-6:
        raise RuntimeError("Anchor and vacancy coincide")
    unit = dvec_mic / dn
    pos_h = pos_a + bond_length * unit

    new_atoms = atoms.copy()
    del new_atoms[v_idx]
    new_atoms.append(Atom("H", position=pos_h))
    return new_atoms


# ============================================================
# QE PWSCF calculator -- GREIGITE (ferrimagnetic A↑↓B + Hubbard U_eff=1.0 eV)
# ============================================================

def inject_hubbard_card(pwi_path, U_eff=1.0, species_with_U=("Fe1", "Fe2")):
    """GREIGITE NEW: append HUBBARD ortho-atomic card (QE 7.2+ syntax) to existing pwi.

    Devey 2009 / Roldan 2016 calibration: U_eff=1.0 eV (Dudarev simplified) on
    Fe-3d for BOTH split sublattices (Fe_tet 8a → Fe1, Fe_oct 16d → Fe2 after
    `normalise_split_labels_in_pwi`). Idempotent: skips if HUBBARD already present.
    """
    pwi_path = Path(pwi_path)
    text = pwi_path.read_text()
    if "HUBBARD" in text:
        return  # idempotent
    lines = ["", "HUBBARD ortho-atomic"]
    for sp in species_with_U:
        lines.append(f"U {sp}-3d {U_eff:.4f}")
    pwi_path.write_text(text + "\n".join(lines) + "\n")


class GreigiteEspresso(Espresso):
    """GREIGITE NEW: Espresso subclass that runs post-writer hooks before pw.x.

    Hooks executed после ASE writes espresso.pwi и before pw.x launch:
      1. `normalise_split_labels_in_pwi` — renames Fe→Fe1, Fe1→Fe2 in ATOMIC_SPECIES
         + ATOMIC_POSITIONS + starting_magnetization keys (ntyp split convention).
      2. `inject_hubbard_card` — appends HUBBARD ortho-atomic + U Fe1-3d / U Fe2-3d.

    Idempotent — re-writes safe (hooks skip if already applied).
    """
    def __init__(self, *args, U_eff=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._U_eff = U_eff

    def write_input(self, atoms, properties=None, system_changes=None):
        super().write_input(atoms, properties=properties, system_changes=system_changes)
        pwi_path = Path(self.directory) / "espresso.pwi"
        if not pwi_path.exists():
            return
        # 1. ntyp-split label normalisation (Fe / Fe1 → Fe1 / Fe2)
        if normalise_split_labels_in_pwi is not None:
            try:
                normalise_split_labels_in_pwi(pwi_path)
            except Exception as e:
                print(f"[WARN] normalise_split_labels_in_pwi failed: {e}", flush=True)
        # 2. HUBBARD card injection (after labels normalised → species names exist)
        try:
            inject_hubbard_card(pwi_path, U_eff=self._U_eff,
                                species_with_U=("Fe1", "Fe2"))
        except Exception as e:
            print(f"[WARN] inject_hubbard_card failed: {e}", flush=True)
        # 3. R1-FIX F5 (CS S1): HUBBARD↔ATOMIC_SPECIES verification gate.
        # Catches silent partial-U bug if apply_greigite_ferri_split emits 3rd
        # Fe-species (QE_BATCH_LESSONS #66 warning).
        import re as _re
        try:
            _text = pwi_path.read_text(encoding="utf-8")
            species_in_pwi = _re.findall(
                r"^\s*(\S+)\s+[\d.]+\s+\S+\.[uU][pP][fF]",
                _text, _re.MULTILINE,
            )
            hubbard_refs = _re.findall(r"U\s+(\S+)-3d", _text)
            missing = [r for r in hubbard_refs if r not in species_in_pwi]
            extra_fe = [s for s in species_in_pwi
                        if s.startswith("Fe") and s not in ("Fe1", "Fe2")]
            if missing or extra_fe:
                raise AssertionError(
                    f"HUBBARD↔ATOMIC_SPECIES mismatch in {pwi_path}: "
                    f"missing_in_species={missing} extra_Fe_unmapped={extra_fe}. "
                    f"This indicates apply_greigite_ferri_split emitted >2 Fe species "
                    f"(e.g. 3rd magmom value from REUSE path or relaxation drift) — "
                    f"U-coverage incomplete. Fix: extend species_with_U to include all Fe-N."
                )
        except FileNotFoundError:
            pass


def make_calc(work_dir, label, kpts=(2, 2, 2),
              ecutwfc=80.0, ecutrho=320.0, mpi_np=1,  # FIXED s150 R3: 60→80, 240→320 (Fe³⁺ HS d⁵ ONCV requirement)
              conv_thr=1.0e-6, mixing_beta=0.05, mixing_mode="plain",
              mixing_ndim=12, mixing_fixed_ns=15,
              diagonalization="david", diago_thr_init=1.0e-4,
              diago_full_acc=True, startingwfc="atomic+random",
              electron_maxstep=500,  # FIXED s150 R3: 200→500 (charge-ordered transition needs more iter)
              disk_io="medium",
              occupations="smearing", smearing="gaussian", degauss=0.005,  # FIXED s150 R3: 0.01→0.005 Ry (Wu 2018 VASP SIGMA=0.05 eV equivalent; old 0.01 too wide для half-metallic gap 0.3-0.4 eV)
              pseudo_files=None, pseudo_dir=None,
              wfc_reuse=False,
              U_eff=1.0):
    """
    GREIGITE FIX: rewritten для ferrimagnetic A↑↓B + Hubbard U_eff=1.0 eV.

    Defaults transfer s142 РЕШЕНИЕ-094 recipe (mack V_Fe AFM+U overnight success):
      - nspin=2 (ferri A↑↓B mandatory; Kwon-Subedi 2011 + Devey 2009)
      - HUBBARD ortho-atomic, U Fe1-3d = U Fe2-3d = 1.0 eV (injected post-writer)
      - mixing_mode='plain', mixing_beta=0.05, mixing_ndim=12, mixing_fixed_ns=15
      - diagonalization='david', diago_thr_init=1e-4, diago_full_acc=True
      - startingwfc='atomic+random' (Roldan 2016 / Devey 2009 standard for greigite)
      - conv_thr=1e-6 Ry (Tier 2 production paper-grade)
      - smearing=gaussian, degauss=0.01 Ry (s142 mack-validated AFM+U recipe, R1-FIX F1)
      - electron_maxstep=200 (generous for ferri+U Fe semicore)
      - ecutwfc 60 Ry / ecutrho 240 Ry (ONCV-SR PBE 4× ratio default)
      - mpirun ALWAYS prefixed (QE GPU mandatory)

    `starting_magnetization` is set via `atoms.set_initial_magnetic_moments()`
    PRIOR к make_calc() call (см. apply_greigite_ferri_split). ASE writes per-species
    block automatically when nspin=2.

    Returns GreigiteEspresso instance — hooks `normalise_split_labels_in_pwi` +
    `inject_hubbard_card` run after each write_input.
    """
    if pseudo_files is None:
        pseudo_files = {"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"}
    if pseudo_dir is None:
        pseudo_dir = PSEUDO_DIR

    # mpirun ALWAYS, even for np=1 (QE GPU init requirement)
    cmd = f"mpirun --allow-run-as-root --bind-to none -np {mpi_np} {PW_BIN}"

    profile = EspressoProfile(
        command=cmd,
        pseudo_dir=pseudo_dir,
    )

    restart_mode = "restart" if wfc_reuse else "from_scratch"

    # GREIGITE FIX: nspin=2 + ferri convention. starting_magnetization picked up
    # automatically by ASE writer from atoms.get_initial_magnetic_moments().
    input_data = {
        "control": {
            "calculation":  "scf",
            "restart_mode": restart_mode,
            "tprnfor":      True,
            "tstress":      False,
            "verbosity":    "high",
            "disk_io":      disk_io,
            "outdir":       str(Path(work_dir) / label / "tmp"),
            "prefix":       label,
        },
        "system": {
            "ecutwfc":     ecutwfc,
            "ecutrho":     ecutrho,
            "occupations": occupations,   # 'smearing' default (greigite ferri+U)
            "nspin":       2,             # GREIGITE FIX: ferrimagnetic A↑↓B
            # HUBBARD card injected POST-WRITER (QE 7.2+ syntax, not in &system)
        },
        "electrons": {
            "conv_thr":         conv_thr,
            "mixing_mode":      mixing_mode,
            "mixing_beta":      mixing_beta,
            "mixing_ndim":      mixing_ndim,         # GREIGITE NEW: РЕШЕНИЕ-094
            "mixing_fixed_ns":  mixing_fixed_ns,     # GREIGITE NEW: РЕШЕНИЕ-094
            "electron_maxstep": electron_maxstep,
            "diagonalization":  diagonalization,
            "diago_thr_init":   diago_thr_init,      # GREIGITE NEW
            "diago_full_acc":   diago_full_acc,      # GREIGITE NEW
            "startingwfc":      startingwfc,         # GREIGITE NEW: atomic+random
        },
    }

    if wfc_reuse:
        # wfc_reuse overrides startingwfc → from saved file
        input_data["electrons"]["startingwfc"] = "file"

    # Smearing block (gaussian narrow degauss for greigite ferri+U)
    if occupations == "smearing":
        input_data["system"]["smearing"] = smearing
        input_data["system"]["degauss"] = degauss

    print(f"[DEBUG make_calc] label={label} U_eff={U_eff} system={input_data['system']}",
          flush=True)

    return GreigiteEspresso(
        profile=profile,
        directory=str(Path(work_dir) / label),
        input_data=input_data,
        pseudopotentials=pseudo_files,
        kpts=tuple(kpts),
        koffset=(0, 0, 0),
        U_eff=U_eff,
    )


# ============================================================
# Pipeline
# ============================================================

def run_greigite(args):
    # GREIGITE FIX: rename run_marcasite → run_greigite (Fd-3m ferri+U pipeline)
    t_start = time.time()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pseudo_files = {
        "Fe": args.pp_fe,
        "S":  args.pp_s,
        "H":  args.pp_h,
    }

    # GREIGITE FIX: unified ferri+U calc kwargs for pristine + endpoints + NEB.
    # Greigite ferri ground state requires nspin=2 + Hubbard EVERYWHERE (otherwise
    # SCF collapses to non-magnetic). Recipe per s142 РЕШЕНИЕ-094 (mack V_Fe AFM+U).
    pristine_calc_kwargs = dict(
        kpts=tuple(args.kpts),
        ecutwfc=args.ecutwfc,
        ecutrho=args.ecutrho,
        mpi_np=args.mpi_np,
        mixing_beta=args.mixing_beta,
        mixing_mode=args.mixing_mode,
        disk_io="medium",
        occupations="smearing",
        smearing=args.smearing,
        degauss=args.degauss,
        pseudo_files=pseudo_files,
        pseudo_dir=args.pseudo_dir,
        wfc_reuse=args.wfc_reuse,
        U_eff=args.U_eff,                          # GREIGITE NEW
    )

    endpoint_calc_kwargs = pristine_calc_kwargs
    # disk_io_neb configurable (s127 plan): 'high' = robust against SIGTERM kill
    # (wfc each SCF to disk, ~1 GB/image × 9 images = ~9 GB workspace);
    # 'medium' = wfc each opt step (less disk, slower recovery on crash);
    # 'low' = no wfc save (smallest disk, full SCF restart on resume — risky).
    neb_calc_kwargs = dict(endpoint_calc_kwargs, disk_io=args.disk_io_neb)

    # GREIGITE FIX: result schema updated for Fd-3m ferri+U
    result = {
        "mineral":   "greigite",
        "spacegroup": "Fd-3m (#227) setting=2",
        "cell_spec": f"conv 56at x {args.repeat[0]}x{args.repeat[1]}x{args.repeat[2]}",
        "code":      "QE PWSCF",
        "method":    f"PBE+U PWFFT, nspin=2 ferri A↑↓B, U_eff={args.U_eff} eV (Dudarev), ecutwfc={args.ecutwfc} Ry",
        "protocol":  "canonical V_Fe (octahedral 16d) + H lateral hop (greigite ferri+U), "
                     "s148 V_Fe pivot после V_S+H deprecated per РЕШЕНИЕ-082 chemistry-signature",
        "kpts":      list(args.kpts),
        "ecutwfc":   args.ecutwfc,
        "ecutrho":   args.ecutrho,
        "mixing_beta": args.mixing_beta,
        "U_eff_eV":  float(args.U_eff),
        "magnetization_init": {
            "Fe_tet_8a":   "+1.0 (fractional → ~+5 μB Fe³⁺ HS d⁵ SCF)",
            "Fe_oct_16d":  "-0.9 (fractional → ~-4 μB Fe²⁺/³⁺ mixed)",
        },
        "n_images":  args.n_images,
        "fmax_neb":  args.fmax_neb,
        "fmax_endpoint": args.fmax_endpoint,
        "decision_ref": "РЕШЕНИЕ-079 (Q-115) + РЕШЕНИЕ-082 chemistry-signature scope (s148) + "
                        "РЕШЕНИЕ-094 AFM+U Tier 1 recipe (s142). "
                        "knowledge/MARCASITE_VSH_ARTIFACT_2026-05-20.md (V_S+H broken for thiospinels).",
        "literature_anchor": "Liu 2021 ACS Omega L-650 V_Fe surface E_a=0.26 eV (extrapolated к octahedral V_Fe in thiospinel).",
    }

    # ---------- build + relax pristine ----------
    # REUSE_FULL: pristine + endA + endB loaded (skip BFGS all 3)
    # REUSE_PRISTINE_ONLY: pristine loaded, endA/endB built fresh via V_Fe picker
    REUSE_FULL = args.reuse_relaxed is not None
    REUSE_PRISTINE_ONLY = (args.reuse_pristine is not None) and not REUSE_FULL
    REUSE = REUSE_FULL  # alias для downstream pristine SP path
    if REUSE_FULL:
        # ASE write() saves QE calc results (nspins, nkpts, eigenvalues, fermi_level, ...)
        # in extended xyz comment line. ASE read() then fails because SinglePointCalculator
        # rejects those non-standard properties: AssertionError: nspins / nkpts / etc.
        # Workaround: monkey-patch SinglePointCalculator.__init__ to filter unsupported props.
        import ase.calculators.singlepoint as _sp
        from ase.calculators.calculator import all_properties as _all_props
        _orig_spc_init = _sp.SinglePointCalculator.__init__
        def _patched_spc_init(self, atoms=None, **results):
            # s150 FIX: kw arg `atoms` not `atoms_obj` для ASE NEB compatibility
            filtered = {k: v for k, v in results.items() if k in _all_props}
            _orig_spc_init(self, atoms, **filtered)
        _sp.SinglePointCalculator.__init__ = _patched_spc_init

        src = Path(args.reuse_relaxed)
        for fn in ("relaxed_pristine.xyz", "relaxed_endA.xyz", "relaxed_endB.xyz"):
            if not (src / fn).exists():
                raise FileNotFoundError(f"--reuse-relaxed missing: {src / fn}")
        print(f"[REUSE] loading relaxed endpoints from {src}", flush=True)
        atoms = read(str(src / "relaxed_pristine.xyz"))
        pre_endA = read(str(src / "relaxed_endA.xyz"))
        pre_endB = read(str(src / "relaxed_endB.xyz"))
        # Detach loaded SinglePointCalculator — мы attach новый QE calc для single-point
        atoms.calc = None
        pre_endA.calc = None
        pre_endB.calc = None
        # Restore original SPC to avoid side-effects elsewhere
        _sp.SinglePointCalculator.__init__ = _orig_spc_init
        print(f"  pristine={atoms.get_chemical_formula()} {len(atoms)}at, "
              f"endA={pre_endA.get_chemical_formula()} {len(pre_endA)}at, "
              f"endB={pre_endB.get_chemical_formula()} {len(pre_endB)}at", flush=True)
        result["reuse_relaxed_from"] = str(src)
        # s150 FIX: ferri split metadata (fe_oct_indices/fe_tet_indices) NOT preserved in
        # xyz round-trip — must re-apply на loaded pristine для V_Fe picker (Phase 3).
        # Mirror логика из REUSE_PRISTINE_ONLY и fresh-build paths ниже.
        if apply_greigite_ferri_split is None:
            raise RuntimeError(
                "apply_greigite_ferri_split not available — afmu_singlepoint_greigite.py "
                "must be on PYTHONPATH."
            )
        atoms, sites, species_order, mag_list = apply_greigite_ferri_split(
            atoms, magmom_tet=+5.0, magmom_oct=-4.0,
        )
        result["fe_tet_indices"] = list(sites["tet"])
        result["fe_oct_indices"] = list(sites["oct"])
        result["species_order_split"] = list(species_order)
        result["magmom_init_list"] = [float(m) for m in mag_list]
        result["magnetism_class"] = "ferrimagnetic_thiospinel_U=1.0_Dudarev"
        print(f"  ferri split re-applied: Fe_tet={len(sites['tet'])}, "
              f"Fe_oct={len(sites['oct'])}", flush=True)
    elif REUSE_PRISTINE_ONLY:
        # NEW s148: reuse only relaxed_pristine.xyz, build endA/endB fresh via V_Fe picker
        import ase.calculators.singlepoint as _sp
        from ase.calculators.calculator import all_properties as _all_props
        _orig_spc_init = _sp.SinglePointCalculator.__init__
        def _patched_spc_init(self, atoms=None, **results):
            # s150 FIX: kw arg `atoms` not `atoms_obj` для ASE NEB compatibility
            filtered = {k: v for k, v in results.items() if k in _all_props}
            _orig_spc_init(self, atoms, **filtered)
        _sp.SinglePointCalculator.__init__ = _patched_spc_init
        try:
            src_pristine = Path(args.reuse_pristine)
            if src_pristine.is_dir():
                src_pristine = src_pristine / "relaxed_pristine.xyz"
            if not src_pristine.exists():
                raise FileNotFoundError(f"--reuse-pristine xyz missing: {src_pristine}")
            print(f"[REUSE-PRISTINE] loading {src_pristine}", flush=True)
            atoms = read(str(src_pristine))
            atoms.calc = None
        finally:
            _sp.SinglePointCalculator.__init__ = _orig_spc_init
        result["reuse_pristine_from"] = str(src_pristine)
        # IMPORTANT: pristine xyz from prior W2 V_S+H run does NOT contain ferri split
        # labels (Fe1/Fe2). Must apply_greigite_ferri_split AFTER load.
        print(f"  {atoms.get_chemical_formula()} {len(atoms)}at — pristine reused, "
              f"applying ferri split now...", flush=True)
        if apply_greigite_ferri_split is None:
            raise RuntimeError(
                "apply_greigite_ferri_split not available — afmu_singlepoint_greigite.py "
                "must be on PYTHONPATH."
            )
        atoms, sites, species_order, mag_list = apply_greigite_ferri_split(
            atoms, magmom_tet=+5.0, magmom_oct=-4.0,  # FIXED s150 R3: absolute μB per QE INPUT_PW.html (|val|>=1)
        )
        result["fe_tet_indices"] = list(sites["tet"])
        result["fe_oct_indices"] = list(sites["oct"])
        result["species_order_split"] = list(species_order)
        result["magmom_init_list"] = [float(m) for m in mag_list]
        result["magnetism_class"] = "ferrimagnetic_thiospinel_U=1.0_Dudarev"
        print(f"  ferri split: Fe_tet={len(sites['tet'])}, Fe_oct={len(sites['oct'])}",
              flush=True)
    else:
        # GREIGITE FIX: Fd-3m conventional cell, then ferri split (tet↑ / oct↓)
        print(f"[1/6] Build greigite Fd-3m 56at x {args.repeat[0]}x{args.repeat[1]}x{args.repeat[2]}",
              flush=True)
        atoms = build_greigite_56at(repeat=tuple(args.repeat))
        # GREIGITE NEW: apply ferrimagnetic A↑↓B init via s142 helper.
        # Signature: (atoms_split, sites, species_order, mag_list)
        if apply_greigite_ferri_split is None:
            raise RuntimeError(
                "apply_greigite_ferri_split not available — afmu_singlepoint_greigite.py "
                "must be on PYTHONPATH (deploy под /workspace/infra/gpu_scripts)."
            )
        # R1-FIX F6 (Chem S1): pass magmom_oct=-1.0 (was -0.9 default) для Devey -32 μB target net magnetization.
        # 8×(+5) + 16×(-4.5) = -32 μB/56at = -4 μB/fu (matches Devey 2009).
        # Settled SCF value with -1.0 init may reach -4.5 μB (Kiejna 2024 Table 1 shows -3.51/+3.56 для PBE+U(1.0)).
        atoms, sites, species_order, mag_list = apply_greigite_ferri_split(
            atoms, magmom_tet=+5.0, magmom_oct=-4.0,  # FIXED s150 R3: absolute μB per QE INPUT_PW.html (|val|>=1)
        )
        result["fe_tet_indices"] = list(sites["tet"])
        result["fe_oct_indices"] = list(sites["oct"])
        result["species_order_split"] = list(species_order)
        result["magmom_init_list"] = [float(m) for m in mag_list]
        # R1-FIX F8 (Chem S4): magnetism class disclaimer для anchor comparison.
        result["magnetism_class"] = "ferrimagnetic_thiospinel_U=1.0_Dudarev"
        result["comparison_caveat"] = (
            "Direct E_a numeric comparison with nspin=1 anchors (pyrite/mack/pent/marcasite) "
            "NOT directly legitimate: greigite Fe³⁺(tet)/Fe²⁺/³⁺(oct) ferrimagnetic +U vs "
            "Fe²⁺ d⁶ LS diamagnetic siblings. Qualitative pathway-pattern comparison OK; "
            "absolute barrier delta interpretation requires SI methodology disclosure."
        )
        print(f"  ferri split: Fe_tet={len(sites['tet'])} (idx {sites['tet'][:3]}...), "
              f"Fe_oct={len(sites['oct'])} (idx {sites['oct'][:3]}...)", flush=True)

    result["n_atoms_pristine"] = len(atoms)
    result["formula_pristine"] = atoms.get_chemical_formula()
    result["cell_A"] = atoms.cell.lengths().tolist()
    print(f"  {atoms.get_chemical_formula()}, {len(atoms)} atoms, "
          f"cell={[f'{x:.3f}' for x in atoms.cell.lengths()]} A", flush=True)

    # PWFFT pristine relax: no Pulay forces => BFGS works cleanly with default maxstep.
    # If forces too high (build basis far from PBE-eq), BFGS may overshoot — fmax 0.03
    # per РЕШЕНИЕ-079 endpoint target is paper-grade and achievable in PWFFT.
    if REUSE_FULL or REUSE_PRISTINE_ONLY:
        print(f"[2/6] Single-point pristine (REUSE — skip BFGS)", flush=True)
    else:
        print(f"[2/6] Relax pristine (BFGS fmax={args.fmax_pristine})", flush=True)
    t0 = time.time()
    label_p = "sp_pristine" if (REUSE_FULL or REUSE_PRISTINE_ONLY) else "relax_pristine"
    atoms.calc = make_calc(work_dir, label_p,
                           conv_thr=args.conv_thr_endpoint,
                           **pristine_calc_kwargs)
    if REUSE_FULL or REUSE_PRISTINE_ONLY:
        try:
            e_pristine = float(atoms.get_potential_energy())
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            conv = True
            nsteps_p = 0
        except RuntimeError as exc:
            raise RuntimeError(f"pristine single-point failed: {exc}") from exc
    else:
        opt = BFGS(atoms, logfile=str(work_dir / "relax_pristine.log"),
                   trajectory=str(work_dir / "relax_pristine.traj"))
        try:
            conv = bool(opt.run(fmax=args.fmax_pristine, steps=args.max_steps_relax))
            e_pristine = float(atoms.get_potential_energy())
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            nsteps_p = int(opt.nsteps)
        except RuntimeError as exc:
            raise RuntimeError(
                f"pristine relax failed (likely SCF non-conv): {exc}. "
                f"Check {work_dir}/relax_pristine/. "
                f"Try mixing_beta=0.1 or check pseudo files."
            ) from exc
    result["E_pristine_eV"] = e_pristine
    result["relax_pristine_steps"] = nsteps_p
    result["relax_pristine_converged"] = conv
    result["relax_pristine_fmax"] = fmax
    result["t_relax_pristine_s"] = time.time() - t0
    # GREIGITE NEW: capture per-atom magnetic moments + tot magnetization
    try:
        magmoms_pristine = atoms.get_magnetic_moments()
        result["magnetic_moments_pristine"] = [float(m) for m in magmoms_pristine]
        result["tot_magnetization_pristine_uB"] = float(np.sum(magmoms_pristine))
        # Site-averaged moments (sanity vs Devey 2009: tet +5, oct -4)
        if "fe_tet_indices" in result:
            tet_m = [float(magmoms_pristine[i]) for i in result["fe_tet_indices"]
                     if i < len(magmoms_pristine)]
            oct_m = [float(magmoms_pristine[i]) for i in result["fe_oct_indices"]
                     if i < len(magmoms_pristine)]
            result["m_Fe_tet_mean_pristine_uB"] = float(np.mean(tet_m)) if tet_m else None
            result["m_Fe_oct_mean_pristine_uB"] = float(np.mean(oct_m)) if oct_m else None
            print(f"  m_Fe_tet_mean={result['m_Fe_tet_mean_pristine_uB']:.3f} μB "
                  f"(target +5.0 Devey 2009), m_Fe_oct_mean="
                  f"{result['m_Fe_oct_mean_pristine_uB']:.3f} μB (target -4.0)",
                  flush=True)
    except Exception as e:
        print(f"  [WARN] magnetic_moments_pristine capture failed: {e}", flush=True)
    print(f"  E={e_pristine:.4f} eV, steps={nsteps_p}, fmax={fmax:.4f}, "
          f"conv={conv}, {result['t_relax_pristine_s']:.0f}s", flush=True)
    write(str(work_dir / "relaxed_pristine.xyz"), atoms)

    if args.skip_endpoints:
        print("[SKIP] endpoints + NEB skipped (--skip-endpoints, pristine-only smoke)",
              flush=True)
        result["skip_endpoints"] = True
        result["skip_neb"] = True
        result["t_total_s"] = time.time() - t_start
        return result

    # ---------- pick canonical V_Fe triple (s148 V_Fe pivot) ----------
    print("[3/6] Pick canonical V_Fe (octahedral 16d) + S_i + S_k triple", flush=True)
    # GREIGITE V_Fe: pick Fe_oct (Wyckoff 16d, sublattice Fe2) only.
    # Octahedral coordination → 6 S neighbours, fe_s_max=2.85 Å (Fe_oct-S ~2.46 nominal).
    sv, si, sk, hop_d, s_neighbours = pick_vfe_oct_and_two_s_neighbours(
        atoms,
        oct_indices=result["fe_oct_indices"],  # set by apply_greigite_ferri_split
        fe_s_max=args.fe_s_max,
        expected_n_s=args.expected_n_s,
    )
    d_sv_si = float(atoms.get_distance(sv, si, mic=True))
    d_sv_sk = float(atoms.get_distance(sv, sk, mic=True))
    print(f"  V_Fe_oct={sv} (Fe2/Fe_oct), S_i={si} (d_FeS={d_sv_si:.3f}), "
          f"S_k={sk} (d_FeS={d_sv_sk:.3f}), hop d_si_sk={hop_d:.3f} A", flush=True)
    result["V_Fe_index"] = int(sv)
    result["V_Fe_sublattice"] = "Fe_oct"
    result["S_i_index"] = int(si)
    result["S_k_index"] = int(sk)
    result["S_neighbours_all"] = [int(x) for x in s_neighbours]
    result["d_VFe_Si_A"] = d_sv_si
    result["d_VFe_Sk_A"] = d_sv_sk
    result["hop_distance_A"] = float(hop_d)

    # ---------- PRE-FLIGHT GATES (s148, mandatory) ----------
    # G1 Wyckoff inequivalence: marc precedent — different orbits для (V_Fe, S, S) triple
    # G2 parity: PRISTINE Fe24 S32 = 24*16+32*6 = 576 e⁻ even — nspin=2 explicit для ferri
    # G3 pocket radius: exclude V_Fe self-distance (vacancy_is_metal=True)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from neb_preflight_gates import run_pre_deploy_gates  # type: ignore
        gates_diag = run_pre_deploy_gates(
            atoms,  # pristine cell с ferri magmoms
            V_S_index=sv,           # actually V_Fe site here
            S_i_index=si,
            S_k_index=sk,
            nspin=2,                # greigite ferrimagnetic — explicit
            metal_symbol="Fe",
            pocket_threshold_A=1.60,
            vacancy_is_metal=True,  # V_Fe RC
        )
        result["preflight_gates"] = gates_diag
    except ImportError as exc:
        print(f"[PRE-FLIGHT WARN] neb_preflight_gates import failed: {exc}. "
              f"Continuing без gate check (NOT recommended).", flush=True)
        result["preflight_gates"] = {"error": str(exc)}

    # ---------- endpoint A ----------
    t0 = time.time()
    if REUSE:
        print(f"[4/6] Single-point endA (REUSE from {args.reuse_relaxed})", flush=True)
        endA = pre_endA
    else:
        print(f"[4/6] Relax endpoint A (V_S={sv}, H on S_i={si}) BFGS fmax={args.fmax_endpoint}",
              flush=True)
        endA = place_h_on_neighbour_s(atoms, v_idx=sv, h_anchor_idx=si)
    label_A = "sp_endA" if REUSE else "relax_endA"
    endA.calc = make_calc(work_dir, label_A,
                          conv_thr=args.conv_thr_endpoint,
                          **endpoint_calc_kwargs)
    if REUSE:
        try:
            e_A = float(endA.get_potential_energy())
            fmaxA = float(np.linalg.norm(endA.get_forces(), axis=1).max())
            convA = True
            nsteps_A = 0
        except RuntimeError as exc:
            raise RuntimeError(f"endA single-point failed: {exc}") from exc
    else:
        optA = BFGS(endA, logfile=str(work_dir / "relax_endA.log"),
                    trajectory=str(work_dir / "relax_endA.traj"))
        try:
            convA = bool(optA.run(fmax=args.fmax_endpoint, steps=args.max_steps_endpoint))
            e_A = float(endA.get_potential_energy())
            fmaxA = float(np.linalg.norm(endA.get_forces(), axis=1).max())
            nsteps_A = int(optA.nsteps)
        except RuntimeError as exc:
            raise RuntimeError(f"endA relax failed: {exc}") from exc
    h_idx_A = len(endA) - 1
    # C1: separate d_H_S_nearest + d_H_Fe_nearest + nearest_nonH identity
    # (chemist MUST-FIX R2 — THE diagnostic для H1/H2/H3 classification)
    syms_A = endA.get_chemical_symbols()
    all_d_A = [(j, syms_A[j], float(endA.get_distance(h_idx_A, j, mic=True)))
               for j in range(len(endA)) if j != h_idx_A]
    all_d_A.sort(key=lambda x: x[2])
    d_H_S_A = min((d for j, sym, d in all_d_A if sym == "S"), default=float("inf"))
    d_H_Fe_A = min((d for j, sym, d in all_d_A if sym == "Fe"), default=float("inf"))
    j_nearest_S_A = next((j for j, sym, d in all_d_A if sym == "S"), -1)
    j_nearest_Fe_A = next((j for j, sym, d in all_d_A if sym == "Fe"), -1)
    nearest_A_j, nearest_A_sym, nearest_A_d = all_d_A[0]
    d_H_A = nearest_A_d  # backward compat
    # H1/H2/H3 endpoint geometry classification (chemist R2 hypothesis matrix)
    endA_S_H_covalent = bool(d_H_S_A < 1.65)   # typical covalent S-H 1.34-1.55 Å
    endA_Fe_H_bond = bool(d_H_Fe_A < 1.85)     # typical Fe-H hydride 1.50-1.75 Å
    if endA_S_H_covalent and not endA_Fe_H_bond:
        endA_geom = "S-H_covalent"
    elif endA_Fe_H_bond and not endA_S_H_covalent:
        endA_geom = "Fe-H_hydride"
    elif endA_S_H_covalent and endA_Fe_H_bond:
        endA_geom = "Fe-H-S_bridge"
    else:
        endA_geom = "broken_no_near_bond"
    result["E_endpointA_eV"] = e_A
    result["relax_endA_steps"] = nsteps_A
    result["endA_converged"] = convA
    result["endA_fmax"] = fmaxA
    result["d_H_nearest_endA"] = d_H_A  # backward compat (any-atom)
    result["d_H_S_nearest_endA"] = d_H_S_A
    result["d_H_Fe_nearest_endA"] = d_H_Fe_A
    result["nearest_S_idx_endA"] = int(j_nearest_S_A)
    result["nearest_Fe_idx_endA"] = int(j_nearest_Fe_A)
    result["nearest_nonH_endA"] = {"idx": int(nearest_A_j), "symbol": nearest_A_sym,
                                    "distance_A": float(nearest_A_d)}
    result["endA_geometry"] = endA_geom
    # GREIGITE NEW: classify nearest-Fe sublattice (tet/oct) — H-anchor preference
    if j_nearest_Fe_A >= 0 and "fe_tet_indices" in result:
        if j_nearest_Fe_A in result["fe_tet_indices"]:
            result["fe_neighbor_type_endA"] = "tet"
        elif j_nearest_Fe_A in result["fe_oct_indices"]:
            result["fe_neighbor_type_endA"] = "oct"
        else:
            result["fe_neighbor_type_endA"] = "unknown"
    else:
        result["fe_neighbor_type_endA"] = "no_Fe_nearby"
    # GREIGITE NEW: capture magnetic moments at endpoint A
    try:
        magmoms_A = endA.get_magnetic_moments()
        result["magnetic_moments_endA"] = [float(m) for m in magmoms_A]
        result["tot_magnetization_endA_uB"] = float(np.sum(magmoms_A))
    except Exception as e:
        print(f"  [WARN] magmoms_endA capture failed: {e}", flush=True)
    result["t_relax_endA_s"] = time.time() - t0
    print(f"  E_A={e_A:.4f} eV, steps={nsteps_A}, fmax={fmaxA:.4f}, conv={convA}", flush=True)
    print(f"  d_H-S={d_H_S_A:.3f} (S#{j_nearest_S_A}), d_H-Fe={d_H_Fe_A:.3f} (Fe#{j_nearest_Fe_A}), "
          f"geom={endA_geom}, fe_neighbor={result['fe_neighbor_type_endA']}", flush=True)
    print(f"  {result['t_relax_endA_s']:.0f}s", flush=True)
    write(str(work_dir / "relaxed_endA.xyz"), endA)

    # ---------- endpoint B ----------
    t0 = time.time()
    if REUSE:
        print(f"[5/6] Single-point endB (REUSE)", flush=True)
        endB = pre_endB
    else:
        print(f"[5/6] Relax endpoint B (V_S={sv}, H on S_k={sk}) BFGS fmax={args.fmax_endpoint}",
              flush=True)
        endB = place_h_on_neighbour_s(atoms, v_idx=sv, h_anchor_idx=sk)
    label_B = "sp_endB" if REUSE else "relax_endB"
    endB.calc = make_calc(work_dir, label_B,
                          conv_thr=args.conv_thr_endpoint,
                          **endpoint_calc_kwargs)
    if REUSE:
        try:
            e_B = float(endB.get_potential_energy())
            fmaxB = float(np.linalg.norm(endB.get_forces(), axis=1).max())
            convB = True
            nsteps_B = 0
        except RuntimeError as exc:
            raise RuntimeError(f"endB single-point failed: {exc}") from exc
    else:
        optB = BFGS(endB, logfile=str(work_dir / "relax_endB.log"),
                    trajectory=str(work_dir / "relax_endB.traj"))
        try:
            convB = bool(optB.run(fmax=args.fmax_endpoint, steps=args.max_steps_endpoint))
            e_B = float(endB.get_potential_energy())
            fmaxB = float(np.linalg.norm(endB.get_forces(), axis=1).max())
            nsteps_B = int(optB.nsteps)
        except RuntimeError as exc:
            raise RuntimeError(f"endB relax failed: {exc}") from exc
    h_idx_B = len(endB) - 1
    # C1 (mirror endA): separate d_H_S/d_H_Fe + geometry classification for endB
    syms_B = endB.get_chemical_symbols()
    all_d_B = [(j, syms_B[j], float(endB.get_distance(h_idx_B, j, mic=True)))
               for j in range(len(endB)) if j != h_idx_B]
    all_d_B.sort(key=lambda x: x[2])
    d_H_S_B = min((d for j, sym, d in all_d_B if sym == "S"), default=float("inf"))
    d_H_Fe_B = min((d for j, sym, d in all_d_B if sym == "Fe"), default=float("inf"))
    j_nearest_S_B = next((j for j, sym, d in all_d_B if sym == "S"), -1)
    j_nearest_Fe_B = next((j for j, sym, d in all_d_B if sym == "Fe"), -1)
    nearest_B_j, nearest_B_sym, nearest_B_d = all_d_B[0]
    d_H_B = nearest_B_d  # backward compat
    endB_S_H_covalent = bool(d_H_S_B < 1.65)
    endB_Fe_H_bond = bool(d_H_Fe_B < 1.85)
    if endB_S_H_covalent and not endB_Fe_H_bond:
        endB_geom = "S-H_covalent"
    elif endB_Fe_H_bond and not endB_S_H_covalent:
        endB_geom = "Fe-H_hydride"
    elif endB_S_H_covalent and endB_Fe_H_bond:
        endB_geom = "Fe-H-S_bridge"
    else:
        endB_geom = "broken_no_near_bond"
    result["E_endpointB_eV"] = e_B
    result["relax_endB_steps"] = nsteps_B
    result["endB_converged"] = convB
    result["endB_fmax"] = fmaxB
    result["d_H_nearest_endB"] = d_H_B
    result["d_H_S_nearest_endB"] = d_H_S_B
    result["d_H_Fe_nearest_endB"] = d_H_Fe_B
    result["nearest_S_idx_endB"] = int(j_nearest_S_B)
    result["nearest_Fe_idx_endB"] = int(j_nearest_Fe_B)
    result["nearest_nonH_endB"] = {"idx": int(nearest_B_j), "symbol": nearest_B_sym,
                                    "distance_A": float(nearest_B_d)}
    result["endB_geometry"] = endB_geom
    # GREIGITE NEW: classify nearest-Fe sublattice for endpoint B
    if j_nearest_Fe_B >= 0 and "fe_tet_indices" in result:
        if j_nearest_Fe_B in result["fe_tet_indices"]:
            result["fe_neighbor_type_endB"] = "tet"
        elif j_nearest_Fe_B in result["fe_oct_indices"]:
            result["fe_neighbor_type_endB"] = "oct"
        else:
            result["fe_neighbor_type_endB"] = "unknown"
    else:
        result["fe_neighbor_type_endB"] = "no_Fe_nearby"
    # GREIGITE NEW: capture magnetic moments at endpoint B
    try:
        magmoms_B = endB.get_magnetic_moments()
        result["magnetic_moments_endB"] = [float(m) for m in magmoms_B]
        result["tot_magnetization_endB_uB"] = float(np.sum(magmoms_B))
    except Exception as e:
        print(f"  [WARN] magmoms_endB capture failed: {e}", flush=True)
    result["t_relax_endB_s"] = time.time() - t0
    print(f"  E_B={e_B:.4f} eV, steps={nsteps_B}, fmax={fmaxB:.4f}, conv={convB}", flush=True)
    print(f"  d_H-S={d_H_S_B:.3f} (S#{j_nearest_S_B}), d_H-Fe={d_H_Fe_B:.3f} (Fe#{j_nearest_Fe_B}), "
          f"geom={endB_geom}, fe_neighbor={result['fe_neighbor_type_endB']}", flush=True)
    print(f"  {result['t_relax_endB_s']:.0f}s", flush=True)
    write(str(work_dir / "relaxed_endB.xyz"), endB)

    if len(endA) != len(endB):
        raise RuntimeError(f"endA/endB atom-count mismatch: {len(endA)} vs {len(endB)}")

    dE_endpoints = abs(e_A - e_B)
    result["dE_endpoints_eV"] = dE_endpoints
    print(f"  |E_A-E_B|={dE_endpoints:.4f} eV (should be ~0 for sym hop)",
          flush=True)
    if dE_endpoints > 0.05:
        warn = (f"WARNING endpoints asymmetric: |dE|={dE_endpoints:.4f} eV > 0.05 "
                f"-> possible wrong triple / SCF drift / local-min issue")
        print(f"  {warn}", flush=True)
        result["endpoints_symmetric"] = False
        result["endpoints_warning"] = warn
    else:
        result["endpoints_symmetric"] = True

    # ============================================================
    # C2: Test A endpoint sanity gate (chemist MUST-FIX R2)
    # Detects V_S+H trap signature (s132 mack precedent: same-basin endpoints)
    # ============================================================
    h_disp = float(endA.get_distance(h_idx_A, h_idx_B,
                                      mic=False))  # endA H pos vs endB H pos
    # MIC: compute displacement via current cells
    pos_h_A = endA.positions[h_idx_A]
    pos_h_B = endB.positions[h_idx_B]
    dvec = pos_h_B - pos_h_A
    cell_arr = endA.cell.array
    fc = np.linalg.solve(cell_arr.T, dvec)
    fc -= np.round(fc)
    h_disp_mic = float(np.linalg.norm(cell_arr.T @ fc))
    result["h_displacement_endA_to_endB_mic_A"] = h_disp_mic
    # R1-FIX F4 (Phys M2): SCF noise floor formula bug.
    # conv_thr in QE is total-energy noise (Ry), NOT per-electron. Multiplying by
    # n_elec_est over-estimated noise by 3 orders of magnitude (5e-3 eV vs real ~1.4e-5 eV)
    # → Test A became BLIND к same-basin trap. Drop the × n_elec multiplier.
    scf_noise_floor_eV = float(args.conv_thr_endpoint * 13.6056)  # Ry → eV (total-E noise)
    result["scf_noise_floor_estimate_eV"] = scf_noise_floor_eV
    same_basin_risk = bool(
        (dE_endpoints < 5 * scf_noise_floor_eV) and (h_disp_mic < 1.0)
    )
    result["test_A_same_basin_risk"] = same_basin_risk
    nearest_match = (nearest_A_sym == nearest_B_sym)
    if nearest_A_sym == "Fe" and nearest_B_sym == "Fe":
        nearest_class = "BOTH_Fe_attraction"  # H1 confirmed pattern
    elif nearest_A_sym == "S" and nearest_B_sym == "S":
        nearest_class = "BOTH_S_anchor"       # H2 confirmed pattern
    else:
        nearest_class = "ASYMMETRIC_bridge_or_broken"
    result["test_A_nearest_class"] = nearest_class
    print(f"  [Test A] h_disp_mic={h_disp_mic:.3f} Å (must >= 1.0 for symmetric hop)",
          flush=True)
    print(f"  [Test A] nearest_class={nearest_class} (A:{nearest_A_sym} B:{nearest_B_sym})",
          flush=True)
    if same_basin_risk:
        warn2 = (f"CRITICAL Test A: same-basin trap signature "
                 f"(dE<5×noise={5 * scf_noise_floor_eV:.2e} eV AND h_disp<1.0 Å). "
                 f"Endpoints likely collapsed to same well. s132 mack/pent precedent.")
        print(f"  {warn2}", flush=True)
        result["test_A_warning"] = warn2
        result["paper_quotable"] = None  # block paper-quotable claim
    # H1/H2/H3 verdict combining endA + endB geometry
    if endA_geom == "S-H_covalent" and endB_geom == "S-H_covalent":
        hypothesis_verdict = "H2_or_H1neg: clean S-H both endpoints (MLIP failure OR S-H is true endpoint)"
    elif endA_geom == "Fe-H_hydride" and endB_geom == "Fe-H_hydride":
        hypothesis_verdict = "H1: Fe-H both endpoints (genuine Fe-attraction)"
    elif endA_geom == "Fe-H-S_bridge" and endB_geom == "Fe-H-S_bridge":
        hypothesis_verdict = "H3: bridge geometry stable both endpoints"
    else:
        hypothesis_verdict = f"asymmetric/mixed: A={endA_geom}, B={endB_geom}"
    result["hypothesis_verdict"] = hypothesis_verdict
    print(f"  [Hypothesis] {hypothesis_verdict}", flush=True)

    if args.skip_neb:
        print("[SKIP] NEB skipped (--skip-neb, endpoints-only smoke)", flush=True)
        result["skip_neb"] = True
        result["t_total_s"] = time.time() - t_start
        return result

    # C3: defensive check — endA.calc и endB.calc должны быть set before NEB
    # (R1 CS finding: endB calc может detach после prewrap copy)
    if endA.calc is None:
        raise RuntimeError("endA.calc is None before NEB — calculator was detached")
    if endB.calc is None:
        raise RuntimeError("endB.calc is None before NEB — calculator was detached after prewrap")

    # ---------- ASE issue #1130 fix: pre-wrap endB relative to endA ----------
    # ASE pre-IDPP linear interp does NOT apply minimum-image для PBC.
    # Atoms whose endpoint position wraps across cell boundary get linearly
    # interpolated через whole cell → atom overlaps → broken initial NEB path.
    # Fix: pre-wrap endB so naive interp is already minimum-image.
    if args.idpp_prewrap:
        endB_unwrapped, n_unwrapped, prewrap_summary = prewrap_endpoint_for_idpp(
            endA, endB, label="endB"
        )
        result["idpp_prewrap_n_unwrapped"] = int(n_unwrapped)
        result["idpp_prewrap_summary"] = prewrap_summary
        if n_unwrapped > 0:
            # CRITICAL: ASE Atoms.copy() does NOT preserve calc, but NEB.get_forces()
            # требует endB.calc для FIRE iterations. Restore calc reference (energy
            # invariant under integer lattice translation, cached results valid).
            endB_unwrapped.calc = endB.calc
            endB = endB_unwrapped
    else:
        result["idpp_prewrap_n_unwrapped"] = 0
        result["idpp_prewrap_summary"] = {"n_unwrapped": 0, "disabled": True}
        print("[IDPP-PREWRAP] disabled via --no-idpp-prewrap (legacy)", flush=True)

    # ---------- CI-NEB ----------
    t0 = time.time()
    # s150 FIX: cleanup sp_*/tmp wfc — single-point reference results already in
    # *.pwo files, tmp wfc not needed for Phase 6. Saves ~25 GB на 50 GB disk
    # (greig W2 hit disk-full на image_06 1st run без этой очистки).
    import shutil as _shutil
    for sp_subdir in ("sp_pristine", "sp_endA", "sp_endB"):
        tmp_path = work_dir / sp_subdir / "tmp"
        if tmp_path.exists():
            try:
                size_mb = sum(f.stat().st_size for f in tmp_path.rglob('*') if f.is_file()) / 1024**2
                _shutil.rmtree(tmp_path)
                print(f"[CLEANUP-pre-NEB] removed {tmp_path} ({size_mb:.0f} MB)", flush=True)
            except Exception as e:
                print(f"[CLEANUP-pre-NEB] WARN failed to remove {tmp_path}: {e}", flush=True)
    print(f"[6/6] CI-NEB ({args.n_images} images, IDPP, FIRE, "
          f"fmax={args.fmax_neb}, k={args.k_spring})", flush=True)
    n_intermediate = args.n_images - 2
    images = [endA]
    for i in range(n_intermediate):
        img = endA.copy()
        img.calc = make_calc(work_dir, f"image_{i+1:02d}",
                             conv_thr=args.conv_thr_neb,
                             **neb_calc_kwargs)
        images.append(img)
    images.append(endB)

    neb, neb_policy = make_neb_band(images, args)
    result.update(neb_policy)
    try:
        neb.interpolate("idpp")
    except Exception as e:
        print(f"  IDPP failed ({e}), fallback linear", flush=True)
        neb.interpolate()

    min_interatomic = float("inf")
    for img in images:
        dmat = img.get_all_distances(mic=True)
        np.fill_diagonal(dmat, np.inf)
        mi = float(dmat.min())
        if mi < min_interatomic:
            min_interatomic = mi
    result["min_interatomic_post_idpp_A"] = min_interatomic
    if min_interatomic < 0.8:
        raise RuntimeError(
            f"IDPP produced overlapping atoms: min_d={min_interatomic:.3f} A < 0.8. "
            f"Retry with linear interp or adjust endpoints."
        )
    print(f"  min interatomic dist (post-IDPP, all images) = "
          f"{min_interatomic:.3f} A", flush=True)

    # Physicist condition #2: sanity check initial path E_a after IDPP, before FIRE.
    # If broken (atom overlap, wrong basin), E_a will be very high (>5 eV), waste hr.
    # ASE issue #1130 case: pre-prewrap gave 28 eV; post-prewrap should be <5 eV.
    print("[INITIAL PATH SANITY] computing E_a from IDPP-interpolated images...",
          flush=True)
    initial_energies = []
    initial_energies.append(e_A)  # endA
    for i, img in enumerate(images[1:-1]):
        try:
            initial_energies.append(float(img.get_potential_energy()))
            print(f"  image {i+1:02d}: E={initial_energies[-1]:.4f} eV", flush=True)
        except RuntimeError as exc:
            raise RuntimeError(f"initial SCF on image {i+1} failed: {exc}") from exc
    initial_energies.append(e_B)  # endB
    e_a_initial = max(initial_energies) - e_A
    result["E_a_initial_post_idpp_eV"] = float(e_a_initial)
    print(f"[INITIAL PATH SANITY] E_a (post-IDPP, pre-FIRE) = "
          f"{e_a_initial:.3f} eV", flush=True)
    if e_a_initial > 5.0:
        raise RuntimeError(
            f"INITIAL PATH SANITY FAIL: E_a (post-IDPP) = {e_a_initial:.3f} eV "
            f">5 eV threshold. Likely broken initial path (atom overlap, wrong "
            f"basin, or PBC wrap not handled). Check IDPP-prewrap status. "
            f"Aborting FIRE — would waste hours on bad path."
        )
    print(f"  → SANE (< 5 eV), proceeding to FIRE", flush=True)

    opt_neb = FIRE(neb, logfile=str(work_dir / "neb.log"),
                   trajectory=str(work_dir / "neb.traj"))
    try:
        neb_conv = bool(opt_neb.run(fmax=args.fmax_neb, steps=args.max_steps_neb))
        energies = [float(img.get_potential_energy()) for img in images]
    except RuntimeError as exc:
        raise RuntimeError(
            f"NEB run/harvest failed after {opt_neb.nsteps} steps: {exc}"
        ) from exc

    e_ref = energies[0]
    rel = [e - e_ref for e in energies]
    e_a = max(rel)
    e_rxn = rel[-1]

    max_idx = int(np.argmax(energies))
    try:
        fci = images[max_idx].get_forces()
        fmax_ci = float(np.linalg.norm(fci, axis=1).max())
    except Exception:
        fmax_ci = float("nan")

    result["neb_k_spring"] = float(args.k_spring)
    result.update(neb_policy)
    result["neb_converged"] = neb_conv
    result["neb_steps"] = int(opt_neb.nsteps)
    result["neb_final_fmax_CI"] = fmax_ci
    result["neb_energies_rel_eV"] = rel
    result["E_a_eV"] = float(e_a)
    result["E_a_paper_quotable"] = float(e_a) if neb_conv else None
    result["E_rxn_eV"] = float(e_rxn)
    result["t_neb_s"] = time.time() - t0
    print(f"  E_a={e_a:.4f} eV (quotable={result['E_a_paper_quotable'] is not None}), "
          f"E_rxn={e_rxn:.4f} eV, steps={opt_neb.nsteps}, conv={neb_conv}, "
          f"fmax_CI={fmax_ci:.4f}, {result['t_neb_s']:.0f}s", flush=True)

    for k, img in enumerate(images):
        write(str(work_dir / f"final_{k:02d}.xyz"), img)

    # ---------- cross-refs ----------
    # GREIGITE FIX: sibling refs updated. No direct DFT-NEB V_S+H literature anchor
    # exists для greigite (Roldan 2016 did H2O molecular adsorption, not V_S hop).
    # Cross-link к marc/pyr siblings (different magnetism class — diamagnetic vs ferri).
    result["sibling_anchors"] = {
        "pyrite_E_a_QE_96at_paper_eV": 0.0946,
        "marc_E_a_QE_96at_paper_eV":   "see s133/s134 output",
        "note": ("Pyr/marc are diamagnetic LS Fe²⁺ d⁶ semiconductors (nspin=1). "
                 "Greigite is ferrimagnetic mixed-valence inverse-thiospinel "
                 "(nspin=2 + Hubbard U=1.0 eV mandatory). Direct E_a comparison "
                 "across magnetism class requires caution.")
    }
    result["NOTE"] = (
        "Canonical 1-vacancy V_S hop on greigite Fe3S4 (Fd-3m setting=2, "
        "conventional 56-atom cell, Z=8). Ferrimagnetic A↑↓B (Fe_tet 8a ↑ +5 μB / "
        "Fe_oct 16d ↓ -4 μB) + Hubbard U_eff=1.0 eV (Dudarev) on both Fe1-3d + Fe2-3d. "
        "Recipe per s142 РЕШЕНИЕ-094 + Devey 2009 + Roldan-de Leeuw 2016. "
        "Full relax pristine + endpoints (NO FixAtoms). Phase 1: 1x1x1 (56at) smoke; "
        "Phase 2 optional: 2x1x1 = 112 atoms finite-size check."
    )

    # ---------- plot ----------
    try:
        # GREIGITE FIX: PNG name greigite (was marc)
        png = output_dir / "neb_canonical_greigite_56at_qe.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.linspace(0, 1, len(rel))
        ax.plot(x, rel, "bo-", linewidth=2, markersize=7)
        ax.set_xlabel("Reaction coordinate")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(f"Greigite Fe3S4 canonical V_Fe (oct 16d) + H hop (QE PWSCF, ferri+U)\n"
                     f"E_a={e_a:.3f} eV, U={args.U_eff} eV, n_atoms={len(endA)}, conv={neb_conv}")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png, dpi=130)
        plt.close(fig)
        result["png_path"] = str(png)
        print(f"  saved {png}", flush=True)
    except Exception as e:
        print(f"  PNG save failed: {e}", flush=True)

    result["t_total_s"] = time.time() - t_start
    return result


def main():
    parser = argparse.ArgumentParser()
    # GREIGITE FIX: paths + defaults for Fd-3m 56at ferri+U workflow
    parser.add_argument("--work-dir",   default="/workspace/neb_canonical_greigite_56at_qe")
    parser.add_argument("--output-dir", default="/workspace/results")
    parser.add_argument("--kpts", type=int, nargs=3, default=[2, 2, 2],
                        help="k-mesh nx ny nz (default 2 2 2 for a~9.88 Å conventional cell)")
    parser.add_argument("--n-images", type=int, default=9,
                        help="total images inc. endpoints")
    # GREIGITE FIX: greigite conventional cell is already 56 atoms — no supercell default
    parser.add_argument("--repeat", type=int, nargs=3, default=[1, 1, 1],
                        help="greigite supercell repeat nx ny nz "
                             "(default 1 1 1 = 56 atoms Phase 1; "
                             "use 2 1 1 = 112 atoms Phase 2 finite-size check)")
    # GREIGITE FIX: paper-grade fmax tightened for ferri+U Tier 2 (РЕШЕНИЕ-094)
    parser.add_argument("--fmax-pristine", type=float, default=0.05,
                        help="pristine relax fmax (eV/A) — greigite ferri+U Tier 2 default")
    parser.add_argument("--fmax-endpoint", type=float, default=0.05,
                        help="endpoint relax fmax (eV/A) — R1-FIX F7 (Chem S3): 0.03→0.05 "
                             "AFM+U thiospinel plateaus 0.1-0.3 typically, 0.05 paper-grade "
                             "per Consilium B marcasite precedent + LITERATURE_FINDINGS line 78. "
                             "Tighten 0.03 only если smoke shows tight convergence (fmax<0.1 by iter 20).")
    parser.add_argument("--fmax-neb", type=float, default=0.05,
                        help="NEB fmax (eV/A) — РЕШЕНИЕ-079: 0.05")
    parser.add_argument("--max-steps-relax",    type=int, default=200)
    parser.add_argument("--max-steps-endpoint", type=int, default=500)
    parser.add_argument("--max-steps-neb",      type=int, default=500)
    parser.add_argument("--k-spring", type=float, default=0.3,
                        help="NEB spring constant (B3 default 0.3)")
    parser.add_argument("--dyneb", dest="dyneb", action="store_true", default=True,
                        help="Use ASE DyNEB dynamic relaxation for selective image updates (default)")
    parser.add_argument("--no-dyneb", dest="dyneb", action="store_false",
                        help="Use standard ASE NEB; keep for A/B baseline comparisons")
    parser.add_argument("--dyneb-scale-fmax", type=float, default=0.0,
                        help="ASE DyNEB scaled convergence factor. Default 0.0 keeps the same "
                             "fmax threshold for all images; tune only after a reference run.")
    # hop_mode default = 'diagonal' для apples-to-apples с MACE s122 +
    # ABACUS sibling (оба default 'diagonal' через args.mack_hop_mode).
    # Q115 protocol §2.1 wording "nearest only" inconsistent с actual s122 finding
    # (V_S=132,S_i=130,S_k=134 — это diagonal mode result). Verify w/ Q115 author later.
    parser.add_argument("--hop-mode", default="diagonal",
                        choices=["nearest", "diagonal", "antipodal"],
                        help="S-S hop topology (default diagonal — apples-to-apples "
                             "MACE s122 / ABACUS sibling)")
    # C4: multi-start H anchor for robustness check (chemist R2)
    # 'S' = default (H placed near S anchor); 'Fe' = H placed near Fe (V_S adjacent);
    # 'mid' = H at midpoint of V_S pocket. Run twice with different anchors
    # to check if endpoint geometry is robust vs initial-condition dependent.
    parser.add_argument("--h-anchor-mode", default="S",
                        choices=["S", "Fe", "mid"],
                        help="initial H placement mode for endpoint relax. "
                             "S=near S anchor (default, S-H covalent start); "
                             "Fe=near V_S-adjacent Fe atom (Fe-H hydride start); "
                             "mid=midpoint of V_S pocket. R2 multi-start robustness check.")
    # QE-specific
    parser.add_argument("--ecutwfc", type=float, default=80.0,
                        help="plane-wave cutoff Ry (FIXED s150 R3: 60→80 для Fe³⁺ HS d⁵ ONCV)")
    parser.add_argument("--ecutrho", type=float, default=320.0,
                        help="density cutoff Ry (FIXED s150 R3: 240→320, 4*ecutwfc ONCV-SR PBE)")
    # GREIGITE FIX: s142 РЕШЕНИЕ-094 recipe — plain mixing β=0.05 + ndim=12 + fixed_ns=15
    parser.add_argument("--mixing-beta", type=float, default=0.05,
                        help="electron mixing (0.05 plain — s142 РЕШЕНИЕ-094 greigite ferri+U; "
                             "0.2 default too aggressive for charged ferri Fe³⁺ HS")
    parser.add_argument("--mixing-mode", default="plain",
                        choices=["plain", "TF", "local-TF"],
                        help="electron mixing mode (plain=Broyden — s142 transferable recipe; "
                             "local-TF if endpoint SCF stalls)")
    # R1-FIX F1 (Chem M1): degauss 0.0015→0.01 Ry per s142 mack-validated РЕШЕНИЕ-094.
    # Roldan VASP ISMEAR=-5 (tetrahedron) ≠ QE gaussian — 0.0015 Ry too narrow для
    # ferri+U half-metal SCF на sparse k-mesh. s142 mack overnight success used 0.01.
    parser.add_argument("--smearing", default="gaussian",
                        choices=["gaussian", "mp", "mv", "fd"],
                        help="smearing type (gaussian default; mv/mp would smear greigite "
                             "PBE gap ~0.5-0.8 eV — avoid)")
    parser.add_argument("--degauss", type=float, default=0.005,
                        help="smearing width Ry (FIXED s150 R3: 0.01→0.005, ≡ 68 meV — "
                             "Wu 2018 VASP ISMEAR=0/SIGMA=0.05 eV equivalent. 0.01 was too "
                             "wide для half-metallic gap 0.3-0.4 eV (Kiejna 2024) — smeared gap; "
                             "0.001 too aggressive — SCF stiffness)")
    # GREIGITE FIX: conv_thr=1e-6 Ry Tier 2 production (Roldan/Devey precedent)
    parser.add_argument("--conv-thr-endpoint", type=float, default=1.0e-6,
                        help="SCF threshold for endpoint relax (Tier 2 paper-grade; 1e-8 "
                             "too tight для greigite ferri+U — SCF won't reach в 200 steps)")
    parser.add_argument("--conv-thr-neb", type=float, default=1.0e-5,
                        help="SCF threshold for NEB images (slightly relaxed vs endpoint)")
    # GREIGITE NEW: Hubbard U_eff (Dudarev simplified, applied to Fe1-3d + Fe2-3d)
    parser.add_argument("--U-eff", type=float, default=1.0,
                        help="Hubbard U_eff (Dudarev, eV) — 1.0 consensus Devey 2009 + "
                             "Roldan 2016 + 4 papers (Q-115 NOMAD). Applied to Fe1-3d "
                             "(tet) AND Fe2-3d (oct), ortho-atomic projection.")
    parser.add_argument("--pseudo-dir", default=PSEUDO_DIR,
                        help="QE pseudopotential directory (W3 default: /opt/pp/oncv_pbe)")
    # ONCV-SR PBE Pseudo Dojo nc-sr-04 standard — matches Q115 protocol §1.
    # Filenames in tarball are bare element symbols.
    parser.add_argument("--pp-fe", default="Fe.upf",
                        help="Fe ONCV-SR PBE pseudopotential (Pseudo Dojo nc-sr-04)")
    parser.add_argument("--pp-s", default="S.upf",
                        help="S ONCV-SR PBE pseudopotential (Pseudo Dojo nc-sr-04)")
    parser.add_argument("--pp-h", default="H.upf",
                        help="H ONCV-SR PBE pseudopotential (Pseudo Dojo nc-sr-04)")
    default_omp = int(os.environ.get("OMP_NUM_THREADS", 8))
    parser.add_argument("--omp",    type=int, default=default_omp)
    parser.add_argument("--mpi-np", type=int, default=1,
                        help="MPI ranks (always wrapped in mpirun for QE GPU init)")
    # s127 production-deploy fixes
    parser.add_argument("--skip-endpoints", action="store_true",
                        help="Skip endpoint A/B + NEB (pristine-only smoke test). "
                             "Use for fastest sanity check (~2 min A100).")
    parser.add_argument("--skip-neb", action="store_true",
                        help="Skip NEB after endpoint relax (pristine + endpoints "
                             "only). Use for paper-grade smoke (~30-90 min A100).")
    parser.add_argument("--disk-io-neb", default="high",
                        choices=["low", "medium", "high"],
                        help="NEB image disk_io ('high' = wfc each SCF for SIGTERM "
                             "recovery on Vast.ai 9 images × ~1 GB; 'medium' = wfc "
                             "each opt step; 'low' = no save, smallest disk, NOT "
                             "recoverable on crash).")
    parser.add_argument("--wfc-reuse", action="store_true",
                        help="Set restart_mode='restart' + startingwfc='file' for ALL "
                             "calcs (resume from saved wfc files after SIGTERM kill). "
                             "Requires previous run with disk_io>='medium'. ~3-4× SCF "
                             "speedup. WARNING: crashes if no save dir exists.")
    # s128 production reuse flag — skip BFGS, jump to NEB on relaxed endpoints
    parser.add_argument("--reuse-relaxed", default=None,
                        help="Path to dir with relaxed_pristine.xyz, relaxed_endA.xyz, "
                             "relaxed_endB.xyz from prior smoke run. Skip BFGS relax, "
                             "do single-point SCF on each (consistent E ref ~6 min on "
                             "A100), then jump to NEB phase. Saves ~6 hr endpoints "
                             "duplicate work after smoke (--skip-neb).")
    # s148 NEW (V_Fe pivot): pristine-only reuse
    parser.add_argument("--reuse-pristine", default=None,
                        help="Path к relaxed_pristine.xyz (file or dir). Skip pristine "
                             "BFGS only; endA/endB built fresh via V_Fe picker. "
                             "NOTE: pristine XYZ does NOT include ferri split labels — "
                             "apply_greigite_ferri_split runs after load. NEW s148.")
    parser.add_argument("--fe-s-max", type=float, default=2.85,
                        help="Fe_oct-S neighbour cutoff (greigite Fe_oct-S ~2.46 nominal, "
                             "2.85 safe для post-relax thermal expansion).")
    parser.add_argument("--expected-n-s", type=int, default=6,
                        help="Expected S neighbours per V_Fe_oct (greigite octahedral = 6).")
    # ASE GitLab issue #1130 fix: pre-wrap endB endpoint so naive linear
    # interp before IDPP is minimum-image-correct. Prevents H atom (or any
    # wrapping atom) from being interpolated through whole cell. Default ON
    # — empirically required for our pyr 96at NEB (s128 H atom wraps z-axis).
    parser.add_argument("--idpp-prewrap", action="store_true", default=True,
                        help="Pre-wrap endB relative to endA via find_mic before NEB "
                             "(fixes ASE issue #1130). Default ON (paper-grade fix).")
    parser.add_argument("--no-idpp-prewrap", dest="idpp_prewrap",
                        action="store_false",
                        help="Disable IDPP prewrap (legacy s127 behavior — broken "
                             "for atoms wrapped across PBC).")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"]      = str(args.omp)
    os.environ["MKL_NUM_THREADS"]      = str(args.omp)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.omp)
    os.environ.setdefault("OMP_STACKSIZE", "256M")

    # Chemist Smoke #1: verify pseudo files exist BEFORE singleton lock + SCF spawn
    missing_pp = []
    for sym, fn in (("Fe", args.pp_fe), ("S", args.pp_s), ("H", args.pp_h)):
        full = Path(args.pseudo_dir) / fn
        if not full.exists():
            missing_pp.append(f"{sym}: {full}")
    if missing_pp:
        print("[FATAL] Missing pseudopotential files:", flush=True)
        for m in missing_pp:
            print(f"  {m}", flush=True)
        print(f"  Available in {args.pseudo_dir}:", flush=True)
        try:
            for f in sorted(Path(args.pseudo_dir).iterdir()):
                print(f"    {f.name}", flush=True)
        except Exception:
            pass
        sys.exit(3)

    acquire_singleton()
    try:
        # GREIGITE FIX: header banner reflects ferri+U workflow
        print("=" * 70, flush=True)
        print("Canonical V_Fe (oct 16d) + H lateral hop NEB: greigite Fe3S4 56 at "
              "(QE PWSCF, ferri+U, s148 V_Fe pivot)",
              flush=True)
        print(f"pw.x={PW_BIN}", flush=True)
        print(f"pseudo_dir={args.pseudo_dir}", flush=True)
        print(f"pseudos: Fe={args.pp_fe}, S={args.pp_s}, H={args.pp_h}", flush=True)
        print(f"kpts={args.kpts}, ecutwfc={args.ecutwfc} Ry, ecutrho={args.ecutrho} Ry",
              flush=True)
        print(f"n_images={args.n_images}, fmax_neb={args.fmax_neb}, "
              f"k_spring={args.k_spring}", flush=True)
        print(f"fmax_pristine={args.fmax_pristine}, fmax_endpoint={args.fmax_endpoint}",
              flush=True)
        print(f"conv_thr endpoint={args.conv_thr_endpoint}, neb={args.conv_thr_neb}",
              flush=True)
        print(f"mixing_mode={args.mixing_mode} beta={args.mixing_beta} (s142 РЕШЕНИЕ-094)",
              flush=True)
        print(f"U_eff={args.U_eff} eV on Fe1-3d + Fe2-3d (Devey 2009 / Roldan 2016)",
              flush=True)
        print(f"smearing={args.smearing} degauss={args.degauss} Ry "
              f"(≡ {args.degauss*13.6:.3f} eV)", flush=True)
        print(f"omp={args.omp}, mpi_np={args.mpi_np}", flush=True)
        print(f"skip_endpoints={args.skip_endpoints}, skip_neb={args.skip_neb}, "
              f"disk_io_neb={args.disk_io_neb}, wfc_reuse={args.wfc_reuse}", flush=True)
        if args.reuse_relaxed:
            print(f"REUSE_RELAXED={args.reuse_relaxed} — skip BFGS, single-point + NEB",
                  flush=True)
        print(f"idpp_prewrap={args.idpp_prewrap} (ASE issue #1130 fix)", flush=True)
        print(f"РЕШЕНИЕ-079 + РЕШЕНИЕ-094 (s142): nspin=2 ferri A↑↓B + U=1.0 eV",
              flush=True)
        print("=" * 70, flush=True)

        try:
            # GREIGITE FIX: run_marcasite → run_greigite
            res = run_greigite(args)
            status = "ok"
        except Exception as e:
            res = {
                "mineral": "greigite",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            status = "fail"
            print(f"[FAIL] {e}", flush=True)
            print(traceback.format_exc(), flush=True)

        res["status"] = status
        res["t_end_epoch"] = time.time()

        # GREIGITE FIX: output JSON name
        out_json = Path(args.output_dir) / "neb_canonical_greigite_56at_qe.json"
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2, cls=NumpyEncoder)
        print(f"\nSaved {out_json} (status={status})", flush=True)
        sys.exit(0 if status == "ok" else 1)
    finally:
        release_singleton()


if __name__ == "__main__":
    # V_Fe pivot per РЕШЕНИЕ-082 — V_S+H deprecated для greigite. This script
    # uses V_Fe (octahedral 16d) RC + pre-flight gates (G1/G2/G3). No deprecation
    # guard needed for this variant.
    main()
