#!/usr/bin/env python3
"""Step 0 (s158 dimer-v6 prep): EXTENDED-region MACE partial Hessian on the
DFT v5 saddle, to resolve the chemist<->physicist split on the imaginary mode.

The triad [H,S18,S71] DFT Hessian gave a mode 96.8% on S18, 1.2% on H. Two
readings: (chemist) real heavy-atom-gated transfer; (physicist) partial-Hessian
artifact / mode-substitution. DECISIVE TEST: widen the Hessian region to the
whole reactive pocket (H + cavity-S within r_s + nearest Fe). If the imaginary
mode then carries real H amplitude ALONG the S18->S71 axis -> transfer mode
(seed it / chemRC). If it stays heavy-S-gated -> that IS the mechanism (finding).
Also: a soft flat-top + S-compression mode can signal a shallow mu-S-H-S bridge
intermediate the symmetric NEB smeared over (mackinawite lesson, Igor s158).

Runs MACE-MP-0 large on the DFT saddle GEOMETRY (v5_saddle_image4.xyz). $0 on gomer.
Reports per-atom amplitude breakdown of each imaginary mode + H axis projection +
saves eigenvector. NOT a DFT result -- a seed/diagnostic to design the DFT dimer.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse, json, os
import numpy as np
import torch
from ase.io import read, write
from ase.vibrations import Vibrations


def mic_vec(cell, d):
    fc = np.linalg.solve(cell.T, d); fc -= np.round(fc); return cell.T @ fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saddle-xyz", default="/workspace/v5_saddle_image4.xyz")
    ap.add_argument("--h-idx", type=int, default=95)
    ap.add_argument("--s-i", type=int, default=18)
    ap.add_argument("--s-k", type=int, default=71)
    ap.add_argument("--r-s", type=float, default=3.3, help="include S within this of H (Ang)")
    ap.add_argument("--n-fe", type=int, default=2, help="include N nearest Fe to H")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--nfree", type=int, default=2)
    ap.add_argument("--out", default="/workspace/results/pyr_vfe_saddle_freq_mace_ext.json")
    ap.add_argument("--vib-name", default="/workspace/vib_ext/vib")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.vib_name), exist_ok=True)

    atoms = read(args.saddle_xyz)
    cell = atoms.cell.array
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    H = args.h_idx

    # --- build extended region: H + cavity-S within r_s + n_fe nearest Fe ---
    s_within = sorted((float(np.linalg.norm(mic_vec(cell, pos[j] - pos[H]))), j)
                      for j in range(len(atoms)) if syms[j] == "S")
    cavity_S = [j for d, j in s_within if d <= args.r_s]
    fe_sorted = sorted((float(np.linalg.norm(mic_vec(cell, pos[j] - pos[H]))), j)
                       for j in range(len(atoms)) if syms[j] == "Fe")
    near_Fe = [j for d, j in fe_sorted[:args.n_fe]]
    region = sorted(set([H] + cavity_S + near_Fe))
    print(f"H={H}; cavity_S(<= {args.r_s} A)={cavity_S}; nearest Fe={near_Fe}", flush=True)
    print(f"REGION ({len(region)} atoms, {3*len(region)} modes): {region}", flush=True)
    for d, j in s_within[:8]:
        print(f"  S{j}: {d:.3f} A  {'[in]' if j in cavity_S else ''}", flush=True)
    for d, j in fe_sorted[:4]:
        print(f"  Fe{j}: {d:.3f} A  {'[in]' if j in near_Fe else ''}", flush=True)

    # --- MACE calc ---
    from mace.calculators import mace_mp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"MACE-MP-0 large on {dev}", flush=True)
    atoms.calc = mace_mp(model="large", device=dev, default_dtype="float64")

    vib = Vibrations(atoms, indices=region, name=args.vib_name, delta=args.delta, nfree=args.nfree)
    print(f"[vib] {len(region)} atoms x 3 x {args.nfree} = {len(region)*3*args.nfree} MACE evals",
          flush=True)
    vib.run()
    vib.summary()

    freqs = vib.get_frequencies()
    thr = 20.0  # cm-1: below = numerical-zero (translation), not imaginary
    imag_idx = []
    imag_list, real_list, nz = [], [], []
    for k, f in enumerate(freqs):
        f = complex(f)
        is_im = abs(f.imag) > abs(f.real)
        mag = abs(f.imag) if is_im else abs(f.real)
        if mag < thr:
            nz.append(round(mag, 2)); continue
        (imag_list.append(round(abs(f.imag), 2)) or imag_idx.append(k)) if is_im else real_list.append(round(abs(f.real), 2))

    # --- decompose each imaginary mode: per-atom amplitude + H axis projection ---
    axis = mic_vec(cell, pos[args.s_k] - pos[args.s_i]); axis /= np.linalg.norm(axis)
    modes_report = {}
    for k in imag_idx:
        m = np.asarray(vib.get_mode(k))            # (natoms,3), nonzero only in region
        amps = {int(i): float(np.linalg.norm(m[i])) for i in region}
        tot = float(np.sqrt(sum(a*a for a in amps.values()))) or 1.0
        frac = {i: round((a*a)/(tot*tot)*100, 1) for i, a in amps.items()}
        h_vec = m[H]; h_amp = float(np.linalg.norm(h_vec))
        h_axis_proj = float(np.dot(h_vec, axis)) if h_amp > 0 else 0.0
        # dominant atom
        dom = max(amps, key=amps.get)
        modes_report[str(k)] = {
            "freq_cm1_imag": round(abs(complex(freqs[k]).imag), 2),
            "dominant_atom": dom, "dominant_symbol": syms[dom],
            "pct_by_atom_sumsq": dict(sorted(frac.items(), key=lambda kv: -kv[1])),
            "H_amplitude": round(h_amp, 4),
            "H_frac_pct": frac.get(H, 0.0),
            "H_proj_on_S18S71_axis_Ang": round(h_axis_proj, 4),
        }
        write(args.out.replace(".json", f"_imagmode{k}.extxyz"),
              _with_mode(atoms, m))

    verdict = ("INDEX_1" if len(imag_idx) == 1 else
               "MINIMUM" if not imag_idx else f"INDEX_{len(imag_idx)}")
    # interpretation hint for the consilium
    interp = "UNKNOWN"
    if imag_idx:
        r0 = modes_report[str(imag_idx[0])]
        if r0["H_frac_pct"] >= 25:
            interp = "H_PARTICIPATES (transfer-like; triad S-dominance was partial-Hessian artifact -> physicist)"
        elif r0["dominant_symbol"] == "S" and abs(r0["H_proj_on_S18S71_axis_Ang"]) > 0.02:
            interp = "S_GATED_with_H_along_axis (heavy-atom-gated transfer -> chemist)"
        else:
            interp = "S_DOMINATED_H_passive (possible mu-bridge / off-axis; investigate)"

    out = {
        "tool": "pyr_vfe_saddle_freq_mace_ext", "model": "MACE-MP-0 large",
        "saddle_xyz": args.saddle_xyz, "region": region, "n_region": len(region),
        "r_s": args.r_s, "n_fe": args.n_fe,
        "n_imaginary": len(imag_idx), "imaginary_freqs_cm1": imag_list,
        "real_freqs_cm1": real_list, "nearzero_cm1": nz,
        "imaginary_modes": modes_report,
        "verdict": verdict, "interpretation": interp,
        "note": ("Extended-region MACE Hessian on DFT v5 saddle. Decides chemist(S-gated real) "
                 "vs physicist(triad artifact) split + seed for DFT dimer v6. NOT a DFT result."),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nN_IMAG={len(imag_idx)} imag={imag_list} cm-1  verdict={verdict}", flush=True)
    print(f"INTERPRETATION: {interp}", flush=True)
    print(json.dumps(modes_report, indent=2), flush=True)
    print(f"saved {args.out}", flush=True)
    open("/workspace/DONE_saddle_freq_ext", "w").write(verdict + " | " + interp)


def _with_mode(atoms, mode):
    a = atoms.copy(); a.arrays["imag_mode"] = mode; return a


if __name__ == "__main__":
    main()
