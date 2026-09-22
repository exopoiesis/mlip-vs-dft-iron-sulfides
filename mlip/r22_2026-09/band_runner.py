"""Evaluate MLIPs on a fixed DFT band — the like-for-like protocol.

Single points at the DFT-converged geometries. No NEB: the model is not allowed to find its own
path, so any difference is the model's energy surface, not its optimiser. This is what showed
MACE's headline "-2.8 meV" on greigite to be -14.9 meV once the geometry was held fixed.

The first model run doubles as an environment check: MACE-MP-0 large on the greigite channel band
gave 221.1 meV on a known-good local install. A host that reproduces that is sound.

  python3 band_runner.py <band_dir> <dft_ea_meV> <model> [model ...]
"""
import sys, json, time
from pathlib import Path
import numpy as np
from ase.io import read


def load(name):
    if name.startswith("mace"):
        from mace.calculators import mace_mp
        variant = name.split(":", 1)[1] if ":" in name else "large"
        return mace_mp(model=variant, dispersion=False, default_dtype="float64", device="cuda")
    if name.startswith("chgnet"):
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator()
    if name.startswith("orb"):
        # orb-models 0.7.0 moved the ASE calculator out of forcefield.calculator; try the
        # documented locations in order rather than pinning one that may move again.
        from orb_models.forcefield import pretrained
        variant = name.split(":", 1)[1] if ":" in name else "orb_v3_conservative_inf_omat"
        model = getattr(pretrained, variant)(device="cuda")
        last = None
        for mod, cls in (("orb_models.forcefield.forcefield_adapter", "ORBCalculator"),
                         ("orb_models.forcefield.calculator", "ORBCalculator"),
                         ("orb_models.calculator", "ORBCalculator")):
            try:
                m = __import__(mod, fromlist=[cls])
                return getattr(m, cls)(model, device="cuda")
            except Exception as e:                       # noqa: PERF203
                last = e
        raise RuntimeError(f"no ORBCalculator found; last error: {last}")

    if name.startswith("fairchem"):
        # fairchem-core 2.20: registry models only (UMA + OMol/OC25/ODAC25 eSEN).
        # The OMat24 eSEN/eqV2 checkpoints are v1-era raw .pt files and are NOT here.
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        spec = name.split(":", 1)[1] if ":" in name else "uma-s-1p2"
        model, _, task = spec.partition("@")
        predictor = pretrained_mlip.get_predict_unit(model, device="cuda")
        return FAIRChemCalculator(predictor, task_name=task or "omat")
    if name.startswith("sevennet"):
        from sevenn.calculator import SevenNetCalculator
        variant = name.split(":", 1)[1] if ":" in name else "7net-0"
        return SevenNetCalculator(variant, device="cuda")
    raise ValueError(f"unknown model spec: {name}")


def main():
    band_dir, dft_ea = Path(sys.argv[1]), float(sys.argv[2])
    models = sys.argv[3:]
    files = sorted(band_dir.glob("final_0*.xyz")) or sorted(band_dir.glob("*.extxyz"))
    imgs = [read(str(f)) for f in files] if len(files) > 1 else read(str(files[0]), index=":")
    print(f"band: {len(imgs)} images from {band_dir}, {imgs[0].get_chemical_formula()}")

    out = {}
    for spec in models:
        print(f"\n=== {spec} ===", flush=True)
        t0 = time.time()
        try:
            calc = load(spec)
        except Exception as e:
            print(f"  load FAILED: {type(e).__name__}: {str(e)[:200]}")
            out[spec] = {"error": f"{type(e).__name__}: {e}"}
            continue
        print(f"  loaded in {time.time()-t0:.1f} s", flush=True)
        try:
            E, F = [], []
            for im in imgs:
                a = im.copy(); a.calc = calc
                E.append(a.get_potential_energy())
                # Forces were being computed and thrown away. Their projection on the path
                # tangent separates "wrong energy surface" from "wrong gradient", which is the
                # difference between a model that would find the right path and one that would not.
                try:
                    F.append(a.get_forces())
                except Exception:
                    F.append(None)
            E = np.array(E, dtype=float)
        except Exception as e:
            print(f"  evaluation FAILED: {type(e).__name__}: {str(e)[:200]}")
            out[spec] = {"error": f"{type(e).__name__}: {e}"}
            continue

        rel = (E - E[0]) * 1000
        ea = float(rel.max())

        # --- force diagnostics on the fixed path -------------------------------------
        cell = np.array(imgs[0].get_cell())

        def mic(d):
            f = np.linalg.solve(cell.T, d.T).T
            f -= np.round(f)
            return f @ cell

        f_par, f_perp, f_max, f_rms_vs_dft = [], [], [], []
        for i, fi in enumerate(F):
            if fi is None:
                continue
            lo, hi = max(i - 1, 0), min(i + 1, len(imgs) - 1)
            t = mic(imgs[hi].get_positions() - imgs[lo].get_positions())
            n = np.linalg.norm(t)
            t = t / n if n > 0 else t
            par = float(np.sum(fi * t))
            f_par.append(par)
            f_perp.append(float(np.linalg.norm(fi - par * t)))
            f_max.append(float(np.abs(fi).max()))
            try:                      # DFT forces, when the band carries them
                fd = imgs[i].get_forces()
                f_rms_vs_dft.append(float(np.sqrt(np.mean((fi - fd) ** 2))))
            except Exception:
                pass

        rec = dict(profile_meV=[round(float(x), 2) for x in rel],
                   force_parallel_eVA=[round(x, 4) for x in f_par],
                   force_perp_norm_eVA=[round(x, 4) for x in f_perp],
                   force_max_eVA=[round(x, 4) for x in f_max],
                   force_rms_vs_dft_eVA=[round(x, 4) for x in f_rms_vs_dft] or None,
                   E_a_meV=round(ea, 2), diff_vs_dft_meV=round(ea - dft_ea, 2),
                   endpoint_drift_meV=round(float(rel[-1]), 3),
                   mirror_meV=round(float(abs(rel[3] - rel[5])), 3) if len(rel) > 5 else None,
                   monotone_to_saddle=bool(np.all(np.diff(rel[:len(rel)//2 + 1]) > 0)),
                   any_below_endpoint=bool(rel.min() < -1.0))
        out[spec] = rec
        print("  profile: " + "  ".join(f"{x:7.1f}" for x in rel))
        print(f"  E_a = {ea:7.1f} meV   DFT {dft_ea}   diff {ea-dft_ea:+.1f}")
        print(f"  drift {rec['endpoint_drift_meV']:+.2f}  mirror {rec['mirror_meV']}  "
              f"monotone {rec['monotone_to_saddle']}  below-endpoint {rec['any_below_endpoint']}")

    # MERGE, never clobber: the first version wrote a fixed filename, so the Orb pass silently
    # destroyed the MACE/CHGNet/SevenNet/UMA results (recoverable only because the logs were kept).
    dest = Path("/workspace") / f"band_{band_dir.name}.json"
    prev = {}
    if dest.exists():
        try:
            prev = json.loads(dest.read_text())
        except Exception:
            dest.replace(dest.with_suffix(".json.corrupt"))
    prev.update(out)
    dest.write_text(json.dumps(prev, indent=2))
    print(f"\nwrote {dest} ({len(prev)} models total, {len(out)} from this run)")


if __name__ == "__main__":
    main()
