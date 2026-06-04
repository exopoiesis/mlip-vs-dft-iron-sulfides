#!/usr/bin/env python3
"""v6 — UNCONSTRAINED saddle-search (ASE Dimer / Sella) for pyrite V_Fe + S-H.

GOAL: strict paper-grade barrier — remove the v5 FixedLine-constrained caveat by
relaxing to the TRUE unconstrained index-1 TS, seeded by the proton-transfer mode.

WHY dimer/Sella not band: band on this PES floors (perpendicular roll-off of slack
images to off-path S20/S43). A single-point min-mode search has NO intermediate
images -> no roll-off floor -> converges to the true saddle (consilium s158).

SEED (s158 Step 0, extended-region MACE Hessian on the DFT saddle): the reactive
mode is a CLEAN STIFF H-transfer — 1051.8i cm-1, 99.9% H amplitude, H projection on
the S18->S71 axis = -0.98. The triad [H,S18,S71] soft mode (209i/163i) was a
partial-Hessian artifact (mode-substitution). => seed = unit H displacement along
S_i->S_k (chemical RC), which the extended Hessian confirmed. Stiff mode + good seed
-> ASE Dimer (built-in, explicit seed, no install) is robust here; Sella optional.

START: v5 converged saddle image 4 (constrained), constraint REMOVED, all atoms free.
CALC: IDENTICAL to v5 (apples-to-apples): nspin=1, ecut 60/480, gaussian degauss 0.01,
kpts 2x2x2, ONCV PBE, U=0; conv_thr 1e-10 (saddle-search forces). mpirun ALWAYS.

POST gates (consilium proximity-gate): d(H-S18) ~ d(H-S71) +-0.1; nearest non-H in
{S18,S71}; barrier in cross-mineral range (~150-270 meV). If H slides to S20/S43/Fe
or an intermediate appears -> FINDING (mu-S-H-S, mack lesson), report, do NOT silently
accept. Final DFT-extended freq + endpoint-freq + nspin=2 control are SEPARATE follow-ups.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.calculators.espresso import Espresso, EspressoProfile

PW_BIN = os.environ.get("PW_BIN", "pw.x")
PSEUDO_DIR = os.environ.get("ESPRESSO_PSEUDO",
                            os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))
# v5 endpoint energy (relaxed endA == endB, symmetric), for barrier readout. Same calc.
E_ENDA_DEFAULT = -124947.94173  # eV (v5, apples-to-apples)
LOCK_FILE = Path("/workspace/.dimer_pyr_qe_VFe_v6.lock")


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def mic_vec(cell, d):
    fc = np.linalg.solve(cell.T, d); fc -= np.round(fc); return cell.T @ fc


def make_calc(work_dir, label, kpts=(2, 2, 2), conv_thr=1.0e-10, mpi_np=1, nspin=1):
    """IDENTICAL to v5 make_calc (smearing branch); conv_thr 1e-10 for saddle forces."""
    pseudo_files = {"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"}
    cmd = f"mpirun --allow-run-as-root --bind-to none -np {mpi_np} {PW_BIN}"
    profile = EspressoProfile(command=cmd, pseudo_dir=PSEUDO_DIR)
    input_data = {
        "control": {"calculation": "scf", "restart_mode": "from_scratch",
                    "tprnfor": True, "tstress": False, "verbosity": "high",
                    "disk_io": "low", "outdir": str(Path(work_dir) / label / "tmp"),
                    "prefix": label},
        "system": {"ecutwfc": 60.0, "ecutrho": 480.0, "occupations": "smearing",
                   "nspin": nspin, "smearing": "gaussian", "degauss": 0.01},
        "electrons": {"conv_thr": conv_thr, "mixing_mode": "plain", "mixing_beta": 0.3,
                      "electron_maxstep": 200, "diagonalization": "david"},
    }
    print(f"[make_calc] {label} system={input_data['system']} conv_thr={conv_thr}", flush=True)
    return Espresso(profile=profile, directory=str(Path(work_dir) / label),
                    input_data=input_data, pseudopotentials=pseudo_files,
                    kpts=tuple(kpts), koffset=(0, 0, 0))


def n_valence_electrons(atoms):
    val = {"Fe": 16, "S": 6, "H": 1}
    return sum(val[s] for s in atoms.get_chemical_symbols())


def build_seed(atoms, h_idx, s_i, s_k, extended_eigvec=None):
    """Seed = H-transfer direction. Default chemical RC (H along S_i->S_k, MIC).
    If extended_eigvec extxyz given, use its imag_mode array (renormalized)."""
    cell = atoms.cell.array
    seed = np.zeros((len(atoms), 3))
    if extended_eigvec and Path(extended_eigvec).exists():
        ev = read(extended_eigvec)
        m = np.asarray(ev.arrays["imag_mode"])
        n = np.linalg.norm(m)
        if n > 0:
            print(f"[seed] from extended eigenvector {extended_eigvec} (|m|={n:.3f})", flush=True)
            return m / n
    axis = mic_vec(cell, atoms.positions[s_k] - atoms.positions[s_i])
    axis /= np.linalg.norm(axis)
    seed[h_idx] = axis
    print(f"[seed] chemical RC: H#{h_idx} along S{s_i}->S{s_k} dir={np.round(axis,3)}", flush=True)
    return seed


def report_saddle(atoms, h_idx, s_i, s_k, e_enda, label):
    d_i = atoms.get_distance(h_idx, s_i, mic=True)
    d_k = atoms.get_distance(h_idx, s_k, mic=True)
    syms = atoms.get_chemical_symbols()
    cell = atoms.cell.array
    # nearest non-H atom to H
    ds = sorted((float(np.linalg.norm(mic_vec(cell, atoms.positions[j] - atoms.positions[h_idx]))), j)
                for j in range(len(atoms)) if j != h_idx)
    nn_d, nn_j = ds[0]
    e = atoms.get_potential_energy()
    barrier = (e - e_enda) * 1000.0
    # 3 nearest S to H (mu-bridge / off-path detection, chemist s158)
    s_near = sorted((float(np.linalg.norm(mic_vec(cell, atoms.positions[j] - atoms.positions[h_idx]))), j)
                    for j in range(len(atoms)) if syms[j] == "S")[:4]
    # off-path S = nearest S NOT an anchor (e.g. S20/S43)
    offpath_S = [(round(d, 3), j) for d, j in s_near if j not in (s_i, s_k)]
    # mu-bridge: a 3rd S pulled into S-H bonding range (< 2.0 A) besides the two anchors
    third_S_d = offpath_S[0][0] if offpath_S else 99.0
    mu_bridge = third_S_d < 2.0
    print(f"[{label}] E={e:.5f} eV  barrier(vs endA)={barrier:.1f} meV", flush=True)
    print(f"[{label}] d(H-S{s_i})={d_i:.3f}  d(H-S{s_k})={d_k:.3f}  asym={abs(d_i-d_k):.3f} A", flush=True)
    print(f"[{label}] nearest non-H = {syms[nn_j]}#{nn_j} @ {nn_d:.3f} A", flush=True)
    print(f"[{label}] 3 nearest S: {[(round(d,3), j) for d, j in s_near[:3]]}; "
          f"off-path S (non-anchor): {offpath_S[:2]}  3rd_S={third_S_d:.3f} A", flush=True)
    # proximity gate
    gate = {"d_HSi": d_i, "d_HSk": d_k, "asym": abs(d_i - d_k),
            "nearest_nonH": f"{syms[nn_j]}{nn_j}", "nearest_d": nn_d, "barrier_meV": barrier,
            "nearest_S": [[round(d, 3), j] for d, j in s_near[:3]],
            "offpath_S": offpath_S[:2], "third_S_d": third_S_d, "mu_bridge_warn": bool(mu_bridge)}
    sym_ok = abs(d_i - d_k) <= 0.15
    anchor_ok = nn_j in (s_i, s_k) and syms[nn_j] == "S"
    gate["symmetric_ok"] = bool(sym_ok)
    gate["anchor_ok"] = bool(anchor_ok)
    gate["verdict"] = ("DIRECT_TRANSFER_OK" if (sym_ok and anchor_ok and not mu_bridge)
                       else "OFF_PATH_OR_INTERMEDIATE_INVESTIGATE")
    print(f"[{label}] PROXIMITY-GATE: {gate['verdict']} "
          f"(sym_ok={sym_ok}, anchor_ok={anchor_ok}, mu_bridge={mu_bridge})", flush=True)
    if gate["verdict"] != "DIRECT_TRANSFER_OK":
        print(f"[{label}] *** H off direct S{s_i}<->S{s_k} transfer OR 3rd S pulled in "
              f"(d={third_S_d:.3f}<2.0) -- possible mu-S-H-S bridge / Fe-H / off-path "
              f"(mack lesson). FINDING, investigate -- do NOT silently accept barrier.", flush=True)
    return gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saddle-xyz", required=True)
    ap.add_argument("--method", choices=("dimer", "sella"), default="dimer")
    ap.add_argument("--work-dir", default="/workspace/dimer_pyr_VFe_v6")
    ap.add_argument("--output-dir", default="/workspace/dimer_pyr_VFe_v6/results")
    ap.add_argument("--h-idx", type=int, default=95)
    ap.add_argument("--s-i", type=int, default=18)
    ap.add_argument("--s-k", type=int, default=71)
    ap.add_argument("--extended-eigvec", default="",
                    help="optional extxyz with imag_mode array (else chemical RC seed)")
    ap.add_argument("--conv-thr", type=float, default=1.0e-10)
    ap.add_argument("--fmax", type=float, default=0.03)
    ap.add_argument("--kpts", default="2,2,2")
    ap.add_argument("--mpi-np", type=int, default=1)
    ap.add_argument("--nspin", type=int, default=1)
    ap.add_argument("--dimer-sep", type=float, default=0.01, help="dimer separation (Ang)")
    ap.add_argument("--kick", type=float, default=0.05, help="initial displacement along seed (Ang)")
    ap.add_argument("--max-num-rot", type=int, default=6,
                    help="dimer rotations/step (physicist s158: 6 for stable capture on early steps)")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--e-enda", type=float, default=E_ENDA_DEFAULT)
    args = ap.parse_args()

    # singleton guard (with stale-lock detection: steal if owner PID is dead)
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            os.kill(old_pid, 0)  # raises if no such process
            print(f"[LOCK] {LOCK_FILE} held by live PID {old_pid} -- another v6 running. Abort.", flush=True)
            raise SystemExit(3)
        except (ValueError, ProcessLookupError):
            print(f"[LOCK] stale lock (dead/garbage PID) -- stealing.", flush=True)
        except PermissionError:
            print(f"[LOCK] PID alive (PermissionError on kill 0) -- abort.", flush=True)
            raise SystemExit(3)
    LOCK_FILE.write_text(str(os.getpid()))

    try:
        wd = Path(args.work_dir); wd.mkdir(parents=True, exist_ok=True)
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        kpts = tuple(int(x) for x in args.kpts.split(","))

        atoms = read(args.saddle_xyz)
        print(f"[load] {args.saddle_xyz}: {len(atoms)} atoms pbc={atoms.get_pbc()}", flush=True)
        # NO constraints (unconstrained search)
        atoms.set_constraint()

        n_e = n_valence_electrons(atoms)
        print(f"[parity] N_e={n_e} ({'ODD' if n_e % 2 else 'EVEN'}), nspin={args.nspin}, "
              f"smearing -> half-e at E_F OK (s158 non-magnetic).", flush=True)
        print(f"[geom] d(H-S{args.s_i})={atoms.get_distance(args.h_idx,args.s_i,mic=True):.3f}  "
              f"d(H-S{args.s_k})={atoms.get_distance(args.h_idx,args.s_k,mic=True):.3f}", flush=True)

        seed = build_seed(atoms, args.h_idx, args.s_i, args.s_k,
                          args.extended_eigvec or None)

        atoms.calc = make_calc(wd, "dimer", kpts=kpts, conv_thr=args.conv_thr,
                               mpi_np=args.mpi_np, nspin=args.nspin)

        traj = str(wd / "v6.traj")
        if args.method == "sella":
            from sella import Sella  # noqa
            print("[method] Sella (order=1, internal=False)", flush=True)
            dyn = Sella(atoms, order=1, internal=False, trajectory=traj,
                        logfile=str(wd / "sella.log"))
            dyn.run(fmax=args.fmax, steps=args.max_steps)
            final = atoms
        else:
            from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate
            print(f"[method] ASE Dimer (sep={args.dimer_sep}, kick={args.kick}, "
                  f"max_num_rot={args.max_num_rot})", flush=True)
            control = DimerControl(initial_eigenmode_method="displacement",
                                   displacement_method="vector",
                                   logfile=str(wd / "dimer_control.log"),
                                   dimer_separation=args.dimer_sep,
                                   trial_angle=np.pi / 4, max_num_rot=args.max_num_rot)
            d_atoms = MinModeAtoms(atoms, control)
            d_atoms.displace(displacement_vector=seed * args.kick)
            dyn = MinModeTranslate(d_atoms, trajectory=traj, logfile=str(wd / "translate.log"))
            dyn.run(fmax=args.fmax, steps=args.max_steps)
            final = d_atoms.atoms if hasattr(d_atoms, "atoms") else atoms

        write(str(out / "v6_saddle_final.xyz"), final)
        gate = report_saddle(final, args.h_idx, args.s_i, args.s_k, args.e_enda, "FINAL")

        fmax_final = float(np.linalg.norm(final.get_forces(), axis=1).max())
        result = {
            "tool": "dimer_pyr_96at_qe_VFe_v6", "method": args.method,
            "calc": {"nspin": args.nspin, "ecutwfc": 60.0, "ecutrho": 480.0,
                     "smearing": "gaussian", "degauss": 0.01, "kpts": list(kpts),
                     "conv_thr": args.conv_thr, "U": 0},
            "fmax_target": args.fmax, "fmax_final": fmax_final,
            "converged": bool(fmax_final <= args.fmax),
            "e_endA_ref_eV": args.e_enda,
            "seed": ("extended_eigvec" if args.extended_eigvec else "chemical_RC"),
            "proximity_gate": gate,
            "note": ("Unconstrained v6 saddle-search. Barrier = E_saddle - E_endA (v5 ref). "
                     "Paper-grade requires follow-up: DFT-extended freq (index-1+ZPE) at THIS "
                     "saddle + endpoint-freq (dZPE#) + nspin=2 control."),
        }
        (out / "dimer_pyr_96at_qe_VFe_v6.json").write_text(
            json.dumps(result, indent=2, cls=NumpyEncoder))
        print(json.dumps(result, indent=2, cls=NumpyEncoder), flush=True)
        (wd / "DONE_v6").write_text(f"{gate['verdict']} fmax={fmax_final:.4f} "
                                    f"barrier={gate['barrier_meV']:.1f}meV\n")
        print(f"[DONE] {gate['verdict']} fmax={fmax_final:.4f} "
              f"barrier={gate['barrier_meV']:.1f} meV converged={fmax_final<=args.fmax}", flush=True)
    finally:
        try:
            LOCK_FILE.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
