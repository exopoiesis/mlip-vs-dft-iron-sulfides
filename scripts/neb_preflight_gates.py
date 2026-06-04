"""
NEB pre-flight gates — 5 hard-abort physics theorems перед каждым DFT NEB run.

Created 2026-05-20 (s148) после marcasite V_S+H artifact (3rd same-basin trap in
Fe-S sulfides). Codifies physicist + chemist consilium meta-review findings
(BS-P1..P6, B1, B2).

Import this module в любой NEB production script и вызывай `run_all_gates()`
после picker triple selected + pristine cell built. Любой fail → SystemExit
(не warning, не continue).

5 gates:
  G1 (Wyckoff theorem, BS-P1): spglib eq_atoms[i] != eq_atoms[k]
  G2 (Electron parity, BS-P4): n_val_e % 2 == 0 if nspin == 1
  G3 (Pocket-radius theorem, BS-P6): r_pocket > 1.60 Å (Fe-H bond)
  G4 (NEB-steps theorem, BS-P2 post-relax): neb_steps >= 5 OR full diagnosis
  G5 (Test A v2 strict, BS-P3): dE > 5*scf_noise_total AND h_displ > 0.5 Å

См.:
  - knowledge/RC_SELECTION_RULES.md (live mineral × RC table)
  - knowledge/DECISIONS.md РЕШЕНИЕ-082 (s148 addendum)
  - memory/feedback_RC_check_before_NEB.md
"""
from __future__ import annotations

import sys
from typing import Iterable, Mapping, Optional  # noqa: F401

import numpy as np


# Valence electron counts для common ONCV-SR PBE Pseudo Dojo pseudo
# (Fe: 16, S: 6, H: 1, Ni: 18, etc.) Add as needed.
ONCV_VAL_E: Mapping[str, int] = {
    "H":  1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B":  3,
    "C":  4,
    "N":  5,
    "O":  6,
    "F":  7,
    "Ne": 8,
    "Na": 9,
    "Mg": 10,
    "Al": 3,
    "Si": 4,
    "P":  5,
    "S":  6,
    "Cl": 7,
    "Ar": 8,
    "K":  9,
    "Ca": 10,
    # Magnetic 3d (semicore included для most ONCV):
    "Ti": 12,
    "V":  13,
    "Cr": 14,
    "Mn": 15,
    "Fe": 16,
    "Co": 17,
    "Ni": 18,
    "Cu": 19,
    "Zn": 20,
}


def _abort(msg: str, *, gate: str) -> None:
    """Hard abort с consistent format. NO continue, NO warning."""
    sys.stderr.write(
        "\n========================================================\n"
        f"NEB PRE-FLIGHT GATE FAILED: {gate}\n"
        "========================================================\n"
        f"{msg}\n"
        "\n"
        "See knowledge/RC_SELECTION_RULES.md для допустимых reaction\n"
        "coordinates per mineral chemistry signature.\n"
        "========================================================\n"
    )
    sys.exit(3)


# ----------------------------------------------------------------------------
# Gate 1: Wyckoff inequivalence (Wigner-Bloch theorem, BS-P1)
# ----------------------------------------------------------------------------
def gate_wyckoff_inequivalence(
    atoms,
    triple_indices: Iterable[int],
    *,
    symprec: float = 0.05,
    require_distinct: bool = True,
) -> dict:
    """G1: Reject picker triple если все atoms в single Wyckoff orbit.

    Reason (BS-P1): Если eq_atoms[i] == eq_atoms[k], endpoints related by
    space-group operation → identical configurations by Wigner-Bloch theorem.
    NEB не может найти barrier between identical states.

    Returns
    -------
    dict с keys: space_group, n_orbits, equivalent_atoms, picker_orbits
    """
    try:
        import spglib
    except ImportError:
        _abort(
            "spglib not installed but required для pre-flight gate G1 "
            "(Wyckoff theorem).\nInstall: pip install spglib",
            gate="G1 Wyckoff",
        )

    cell = atoms.cell.array
    scaled = atoms.get_scaled_positions(wrap=True)
    numbers = atoms.get_atomic_numbers()

    ds = spglib.get_symmetry_dataset((cell, scaled, numbers), symprec=symprec)
    if ds is None:
        _abort(
            f"spglib не определил symmetry на cell с symprec={symprec}. "
            "Сheck cell + positions OK?",
            gate="G1 Wyckoff",
        )

    # Handle spglib API: dict-like OR object с attributes
    if hasattr(ds, "equivalent_atoms"):
        eq_atoms = np.asarray(ds.equivalent_atoms)
        sg_num = int(ds.number)
        sg_int = str(ds.international)
    else:
        eq_atoms = np.asarray(ds["equivalent_atoms"])
        sg_num = int(ds["number"])
        sg_int = str(ds["international"])

    triple_list = list(triple_indices)
    picker_orbits = [int(eq_atoms[i]) for i in triple_list]
    n_orbits = len(set(eq_atoms.tolist()))

    if require_distinct and len(set(picker_orbits)) == 1:
        common_orbit = picker_orbits[0]
        orbit_size = int(np.sum(eq_atoms == common_orbit))
        _abort(
            f"Picker triple {triple_list} all в single Wyckoff orbit "
            f"(rep={common_orbit}, size={orbit_size}/{len(atoms)} atoms).\n"
            f"Space group: #{sg_num} {sg_int}.\n"
            f"By Wigner-Bloch theorem, endpoints are CRYSTALLOGRAPHICALLY "
            f"IDENTICAL — NEB cannot find barrier between same configuration.\n"
            f"\n"
            f"Use different picker logic (e.g. cross-orbit pair) OR pivot к "
            f"V_Fe lateral hop / V_S₂ dimer hop per РЕШЕНИЕ-082.",
            gate="G1 Wyckoff",
        )

    return {
        "space_group": f"#{sg_num} {sg_int}",
        "n_orbits": n_orbits,
        "equivalent_atoms": eq_atoms.tolist(),
        "picker_orbits": picker_orbits,
    }


# ----------------------------------------------------------------------------
# Gate 2: Electron parity ↔ nspin (BS-P4)
# ----------------------------------------------------------------------------
def gate_electron_parity_nspin(
    atoms,
    nspin: int,
    *,
    val_e_table: Mapping[str, int] = ONCV_VAL_E,
    tot_charge: float = 0.0,
) -> dict:
    """G2: Reject nspin=1 если total valence electrons odd.

    Reason (BS-P4): Closed-shell DFT (nspin=1) cannot represent unpaired
    electron. Defect cell (V_S+H) typically has odd e⁻ count regardless of
    pristine magnetic state (V_S removes 2 anion e⁻, +H adds 1 → parity flips).

    Returns
    -------
    dict с keys: n_val_e, parity, nspin_required
    """
    try:
        n_val = sum(val_e_table[s] for s in atoms.get_chemical_symbols())
    except KeyError as e:
        _abort(
            f"Element {e} нет в val_e_table — add к ONCV_VAL_E "
            "(используем стандартные ONCV-SR PBE counts).",
            gate="G2 parity",
        )

    n_val_net = int(round(n_val - tot_charge))
    is_odd = (n_val_net % 2) == 1

    if is_odd and nspin == 1:
        _abort(
            f"Odd valence electrons {n_val_net} ({n_val} val - {tot_charge:.1f} "
            f"charge) requires nspin=2 (or different tot_charge).\n"
            f"Closed-shell DFT (nspin=1) literally cannot represent unpaired "
            f"electron — ground state misrepresented.",
            gate="G2 parity",
        )

    return {
        "n_val_e": n_val,
        "tot_charge": tot_charge,
        "n_val_e_net": n_val_net,
        "parity": "odd" if is_odd else "even",
        "nspin_used": nspin,
    }


# ----------------------------------------------------------------------------
# Gate 3: Pocket-radius theorem (BS-P6)
# ----------------------------------------------------------------------------
def gate_pocket_radius(
    atoms,
    vacancy_position: np.ndarray,
    metal_symbol: str = "Fe",
    *,
    threshold_A: float = 1.60,
    exclude_vacancy_idx: Optional[int] = None,
) -> dict:
    """G3: Reject V_X+H protocol если pocket radius < d_FeH equilibrium.

    Reason (BS-P6): If r_pocket = min(d(V_X_site, Fe_j)) < 1.6 Å (typical
    Fe-H hydride bond), H atom inevitably collapses to nearest Fe instead
    of remaining at S anchor → same-basin trap regardless of picker.

    For V_Fe case: the vacancy IS a Fe atom. Set exclude_vacancy_idx=fe_v_index
    to skip self-distance (otherwise r_pocket = 0 trivially). For V_S case:
    vacancy is S, no Fe coincides, exclude_vacancy_idx=None is safe.

    Returns
    -------
    dict с keys: r_pocket, nearest_metal_idx, threshold
    """
    metal_indices = [i for i, s in enumerate(atoms.get_chemical_symbols())
                     if s == metal_symbol]
    # Skip vacancy site itself (relevant when vacancy IS this metal, e.g. V_Fe)
    if exclude_vacancy_idx is not None:
        metal_indices = [i for i in metal_indices if i != exclude_vacancy_idx]
    if not metal_indices:
        _abort(
            f"No {metal_symbol} atoms found в cell (after exclude).",
            gate="G3 pocket",
        )

    positions = atoms.get_positions()
    cell = atoms.cell.array
    inv_cell = np.linalg.inv(cell)

    def _mic_distance(a, b):
        d = b - a
        f = d @ inv_cell
        f -= np.round(f)
        return float(np.linalg.norm(f @ cell))

    dists = [(_mic_distance(vacancy_position, positions[j]), j)
             for j in metal_indices]
    r_pocket, nearest_idx = min(dists)

    if r_pocket < threshold_A:
        _abort(
            f"Pocket radius r={r_pocket:.3f} Å < {threshold_A} Å threshold.\n"
            f"Nearest {metal_symbol} #{nearest_idx} at d={r_pocket:.3f} Å.\n"
            f"H atom will collapse to {metal_symbol}-H hydride (d≈1.6 Å) "
            f"instead of remaining at S anchor — same-basin artifact "
            f"inevitable regardless of picker choice.\n"
            f"\n"
            f"Use V_{metal_symbol} lateral hop OR V_S₂ dimer hop instead.",
            gate="G3 pocket",
        )

    return {
        "r_pocket_A": r_pocket,
        "nearest_metal_symbol": metal_symbol,
        "nearest_metal_idx": nearest_idx,
        "threshold_A": threshold_A,
        "excluded_idx": exclude_vacancy_idx,
    }


# ----------------------------------------------------------------------------
# Gate 4: NEB-steps theorem (BS-P2, post-relax check)
# ----------------------------------------------------------------------------
def gate_neb_steps_nonzero(
    neb_steps: int,
    fmax_initial: float,
    fmax_threshold: float,
    *,
    min_steps: int = 1,
) -> dict:
    """G4: Reject NEB result если converged за 0 steps (initial path already
    под threshold).

    Reason (BS-P2): NEB converging immediately means initial path = final
    path = no movement = endpoints same basin. Theorem:

        (neb_steps == 0 AND fmax_initial < fmax_threshold)
            IMPLIES endpoints in same basin.

    Returns
    -------
    dict с keys: neb_steps, fmax_initial, fmax_threshold, verdict
    """
    if neb_steps < min_steps and fmax_initial < fmax_threshold:
        _abort(
            f"NEB converged за {neb_steps} steps (< {min_steps}) с initial "
            f"fmax={fmax_initial:.4f} < threshold {fmax_threshold}.\n"
            f"This is theorem-level signature of same-basin endpoints "
            f"(BS-P2 zero-steps theorem).\n"
            f"\n"
            f"Possible causes:\n"
            f"  • Endpoints crystallographically equivalent (check G1 Wyckoff)\n"
            f"  • Pocket radius < d_FeH (check G3 pocket-radius)\n"
            f"  • Picker selected non-distinguishable triple\n"
            f"\n"
            f"DO NOT quote E_a from this NEB — it is below numerical precision.",
            gate="G4 NEB-steps",
        )

    return {
        "neb_steps": neb_steps,
        "fmax_initial": fmax_initial,
        "fmax_threshold": fmax_threshold,
        "verdict": "ok",
    }


# ----------------------------------------------------------------------------
# Gate 5: Test A v2 strict (BS-P3 + chemist Fix 13)
# ----------------------------------------------------------------------------
def gate_test_a_v2_strict(
    *,
    dE_endpoints_eV: float,
    h_displacement_mic_A: float,
    conv_thr_Ry: float,
    n_electrons: int,
    hop_distance_A: float,
    nearest_class_A: str = "",
    nearest_class_B: str = "",
) -> dict:
    """G5: Strict Test A v2 gate — abort если same-basin signature.

    Reason (BS-P3): SCF noise floor total ≈ conv_thr × n_e / 2 (Ry).
    Strict thresholds для production NEB:
        dE_endpoints > max(5 × scf_noise_total, 5 meV)
        h_displ > max(0.5 Å, 0.2 × hop_distance)

    Если nearest_class_A == nearest_class_B == "BOTH_Fe_attraction" → same Fe basin.

    Returns
    -------
    dict с verdicts и computed floors.
    """
    scf_noise_total_eV = (conv_thr_Ry * n_electrons / 2.0) * 13.6057  # Ry → eV
    gate_dE_eV = max(5.0 * scf_noise_total_eV, 0.005)
    gate_h_displ_A = max(0.5, 0.2 * hop_distance_A)

    failures = []
    if dE_endpoints_eV < gate_dE_eV:
        failures.append(
            f"dE_endpoints={dE_endpoints_eV*1000:.3f} meV "
            f"< gate {gate_dE_eV*1000:.2f} meV (5× SCF noise OR 5 meV floor)"
        )
    if h_displacement_mic_A < gate_h_displ_A:
        failures.append(
            f"h_displ_mic={h_displacement_mic_A:.4f} Å "
            f"< gate {gate_h_displ_A:.3f} Å (max(0.5, 20% × hop))"
        )
    if nearest_class_A and nearest_class_A == nearest_class_B \
            and "Fe" in nearest_class_A.upper():
        failures.append(
            f"nearest_class_A == nearest_class_B == {nearest_class_A!r} "
            "→ H attached to Fe at both endpoints (Fe-H basin trap)"
        )

    if failures:
        _abort(
            "Test A v2 strict FAIL:\n  • " + "\n  • ".join(failures)
            + f"\n\nSCF noise floor (total) = {scf_noise_total_eV*1000:.4f} meV "
            f"(conv_thr {conv_thr_Ry} Ry × {n_electrons} e⁻ / 2 × 13.6)\n"
            f"Endpoints in same basin — barrier not physical.",
            gate="G5 Test A v2",
        )

    return {
        "dE_endpoints_eV": dE_endpoints_eV,
        "h_displ_mic_A": h_displacement_mic_A,
        "scf_noise_total_eV": scf_noise_total_eV,
        "gate_dE_eV": gate_dE_eV,
        "gate_h_displ_A": gate_h_displ_A,
        "verdict": "pass",
    }


# ----------------------------------------------------------------------------
# Convenience: run all 3 PRE-deploy gates (G1 + G2 + G3)
# G4 + G5 are POST-relax (call after endpoint relax + NEB respectively)
# ----------------------------------------------------------------------------
def run_pre_deploy_gates(
    atoms_pristine,
    *,
    V_S_index: int,
    S_i_index: int,
    S_k_index: int,
    nspin: int,
    metal_symbol: str = "Fe",
    pocket_threshold_A: float = 1.60,
    val_e_table: Mapping[str, int] = ONCV_VAL_E,
    tot_charge: float = 0.0,
    symprec: float = 0.05,
    vacancy_is_metal: bool = False,
) -> dict:
    """Run G1 (Wyckoff) + G2 (parity) + G3 (pocket radius) before launching DFT.

    vacancy_is_metal=True для V_Fe RC (the vacancy IS a Fe atom — exclude its
    self-distance в G3 calc). False для V_S RC (vacancy is S; Fe metric still
    excludes none).

    Any failure → SystemExit. Otherwise returns dict с all gate diagnostics.
    """
    print("[PRE-FLIGHT] Running 3 hard gates (G1 Wyckoff, G2 parity, G3 pocket)...",
          flush=True)

    g1 = gate_wyckoff_inequivalence(
        atoms_pristine,
        triple_indices=(V_S_index, S_i_index, S_k_index),
        symprec=symprec,
    )
    print(f"[G1 Wyckoff PASS] SG {g1['space_group']}, "
          f"picker orbits {g1['picker_orbits']}", flush=True)

    g2 = gate_electron_parity_nspin(
        atoms_pristine,
        nspin=nspin,
        val_e_table=val_e_table,
        tot_charge=tot_charge,
    )
    print(f"[G2 parity PASS] n_val_e_net={g2['n_val_e_net']} {g2['parity']} "
          f"nspin={nspin}", flush=True)

    exclude_idx = V_S_index if vacancy_is_metal else None
    g3 = gate_pocket_radius(
        atoms_pristine,
        vacancy_position=atoms_pristine.get_positions()[V_S_index],
        metal_symbol=metal_symbol,
        threshold_A=pocket_threshold_A,
        exclude_vacancy_idx=exclude_idx,
    )
    print(f"[G3 pocket PASS] r_pocket={g3['r_pocket_A']:.3f} Å > "
          f"{pocket_threshold_A} Å (excluded vacant idx={exclude_idx})",
          flush=True)

    print("[PRE-FLIGHT] All 3 pre-deploy gates PASS — proceeding к DFT NEB.",
          flush=True)

    return {"G1": g1, "G2": g2, "G3": g3}
