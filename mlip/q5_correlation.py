#!/usr/bin/env python
"""Q5 (Stanford review): per-image MLIP-vs-DFT band correlation analysis.

For each NEB case with a tabulated DFT band, compares the foundation-MLIP
per-image relative energies (MACE-MP-0 large, CHGNet v0.3.0) against the DFT
band on the SAME endpoints. Computes Pearson R, RMSE, saddle-index agreement,
and a barrier comparison, then classifies the MLIP failure type:

  - energetic        : barrier magnitude wrong but shape/saddle roughly tracked
  - topological      : saddle location wrong / band collapse / sign-flipped
  - energetic+topological : MLIP flat (no barrier at all) -> R~0, no saddle

Data sources are LOCAL JSON / tabulated bands only ($0, no DFT re-run).
Numpy-only (no ASE needed here); DFT bands pre-extracted to tmp/*.json by
tmp/extract_pyr_dft_band.py.

Outputs:
  tmp/q5_summary.json   : machine-readable metrics
  stdout                : human-readable tables + metrics
"""
import json
import os
import sys

import numpy as np

ROOT = "/d/home/ignat/project-third-matter"
# When run from a Windows python the cwd is set by the wrapper; use relative paths
# resolved against the project root if ROOT exists, else cwd.
def P(rel):
    base = ROOT if os.path.isdir(ROOT) else "."
    return os.path.join(base, rel)


def load_mlip_band(path, mineral):
    with open(path) as fh:
        d = json.load(fh)
    run = d["runs"][mineral]
    band_eV = run["neb_energies_rel_eV"]
    return np.array(band_eV, dtype=float)


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.std(a) < 1e-30 or np.std(b) < 1e-30:
        return float("nan")  # one series is flat -> correlation undefined
    return float(np.corrcoef(a, b)[0, 1])


def rmse_meV(dft_meV, mlip_meV):
    dft = np.asarray(dft_meV, float)
    mlip = np.asarray(mlip_meV, float)
    return float(np.sqrt(np.mean((dft - mlip) ** 2)))


def analyze(case_name, dft_meV, mace_eV=None, chgnet_eV=None, note=""):
    n = len(dft_meV)
    dft = np.asarray(dft_meV, float)
    res = {
        "case": case_name,
        "n_images": n,
        "dft_band_meV": [round(x, 3) for x in dft.tolist()],
        "dft_saddle_index": int(np.argmax(dft)),
        "dft_barrier_meV": round(float(dft.max() - dft.min()), 3),
        "note": note,
        "models": {},
    }
    for tag, band_eV in (("mace_mp0_large", mace_eV), ("chgnet_v030", chgnet_eV)):
        if band_eV is None:
            continue
        mlip_meV = np.asarray(band_eV, float) * 1000.0
        R = pearson(dft, mlip_meV)
        rm = rmse_meV(dft, mlip_meV)
        saddle = int(np.argmax(mlip_meV))
        barrier = float(mlip_meV.max() - mlip_meV.min())
        res["models"][tag] = {
            "mlip_band_meV": [round(x, 4) for x in mlip_meV.tolist()],
            "pearson_R": (None if np.isnan(R) else round(R, 4)),
            "rmse_meV": round(rm, 3),
            "mlip_saddle_index": saddle,
            "saddle_matches_dft": (saddle == int(np.argmax(dft))),
            "mlip_barrier_meV": round(barrier, 4),
            "barrier_ratio_mlip_over_dft": (
                round(barrier / res["dft_barrier_meV"], 4)
                if res["dft_barrier_meV"] > 1e-9 else None
            ),
        }
    return res


def classify(case_res):
    """Heuristic classification of failure type per model."""
    out = {}
    dft_barrier = case_res["dft_barrier_meV"]
    for tag, m in case_res["models"].items():
        R = m["pearson_R"]
        ratio = m["barrier_ratio_mlip_over_dft"]
        saddle_ok = m["saddle_matches_dft"]
        label_parts = []
        # MLIP essentially flat (no barrier): barrier << DFT and R undefined/low
        if ratio is not None and ratio < 0.10:
            label_parts.append("energetic+topological (MLIP flat: no barrier, no saddle)")
        else:
            if ratio is not None and (ratio < 0.5 or ratio > 2.0):
                label_parts.append("energetic (barrier magnitude wrong)")
            if not saddle_ok:
                label_parts.append("topological (saddle misplaced)")
            if R is not None and R < 0.5:
                label_parts.append("force-misalignment (low shape correlation)")
            if not label_parts:
                label_parts.append("shape tracked (qualitative agreement)")
        out[tag] = "; ".join(label_parts)
    return out


def main():
    # ---- DFT bands ----
    # Pyrite V_S2 canonical (E_a = 94.6 meV), extracted from neb.traj last 9 frames
    with open(P("tmp/pyr_vs2_dft_band.json")) as fh:
        pyr_vs2 = json.load(fh)
    pyr_vs2_dft_meV = pyr_vs2["rel_to_img0_meV"]

    # Pyrite V_Fe (separate canonical path, informational) extracted analogously
    pyr_vfe_dft_meV = None
    try:
        with open(P("tmp/pyr_vfe_dft_band.json")) as fh:
            pyr_vfe_dft_meV = json.load(fh)["rel_to_img0_meV"]
    except FileNotFoundError:
        pass

    # Greigite V_Fe DFT band -- TABULATED in SI_K (do not recompute), eV referenced to endA
    greig_vfe_dft_eV = [0.0, 0.099, 0.603, 1.58, 1.86, 1.58, 0.603, 0.099, 0.0]
    greig_vfe_dft_meV = [x * 1000.0 for x in greig_vfe_dft_eV]

    # ---- MLIP bands ----
    mace_json = P("results/mlip_canonical/mace_mp0_large_canonical_1vacancy.json")
    chgnet_json = P("results/mlip_canonical/chgnet_v030_canonical_1vacancy.json")

    pyr_mace = load_mlip_band(mace_json, "pyrite")     # V_S vacancy, hop ~3.06 A
    pyr_chg = load_mlip_band(chgnet_json, "pyrite")
    mack_mace = load_mlip_band(mace_json, "mackinawite")
    mack_chg = load_mlip_band(chgnet_json, "mackinawite")

    cases = []

    # CASE 1: pyrite V_S2 -- the primary, fully-comparable case
    c1 = analyze(
        "pyrite_V_S2",
        pyr_vs2_dft_meV,
        mace_eV=pyr_mace,
        chgnet_eV=pyr_chg,
        note="V_S2 sulfur-vacancy H-hop, 96-atom cell, 9 images, symmetric, "
             "DFT saddle img4 = 94.6 meV (QE PBE). MLIP same vacancy/hop topology.",
    )
    c1["failure_classification"] = classify(c1)
    cases.append(c1)

    # CASE 2: greigite V_Fe -- DFT band tabulated; MLIP pristine SP unphysical
    # MACE/CHGNet pristine greigite single-point ~1e10 eV (NOT a usable band) ->
    # correlation NOT applicable; recorded qualitatively only.
    c2 = analyze(
        "greigite_V_Fe",
        greig_vfe_dft_meV,
        mace_eV=None,
        chgnet_eV=None,
        note="DFT band from SI_K (E_a=1.86 eV, saddle img4). Foundation MLIP "
             "pristine greigite single-point energy ~1e10 eV (non-physical) -> "
             "no usable MLIP band; correlation not applicable (qualitative fail).",
    )
    c2["failure_classification"] = {
        "mace_mp0_large": "catastrophic (pristine SP ~1e10 eV, no band)",
        "chgnet_v030": "catastrophic (pristine SP ~1e10 eV, no band)",
    }
    cases.append(c2)

    # CASE 3: mackinawite -- both DFT and MLIP available as a contrast/control.
    # DFT mackinawite V_S band: from MLIP-vs-DFT main paper, mackinawite is the
    # ONE case where MLIP succeeds (low barrier). We do NOT have a per-image DFT
    # mackinawite band locally extracted here; flag as pending so we don't invent.
    c3 = {
        "case": "mackinawite_V_S",
        "n_images": 9,
        "note": "MLIP bands available (MACE/CHGNet both ~flat, E_a<0.0003 meV), "
                "but per-image DFT mackinawite band not extracted in this pass "
                "(no local neb.traj located). [per-image DFT mackinawite band: "
                "extract pending].",
        "mace_band_meV": [round(x * 1000.0, 4) for x in mack_mace.tolist()],
        "chgnet_band_meV": [round(x * 1000.0, 4) for x in mack_chg.tolist()],
    }
    cases.append(c3)

    # CASE 4 (informational): pyrite V_Fe DFT band (separate path) -- no matched
    # MLIP V_Fe band in canonical JSON (canonical pyrite MLIP = V_S), so recorded
    # as DFT-only reference.
    if pyr_vfe_dft_meV is not None:
        cases.append({
            "case": "pyrite_V_Fe_DFT_only",
            "n_images": 9,
            "dft_band_meV": [round(x, 3) for x in pyr_vfe_dft_meV],
            "dft_saddle_index": int(np.argmax(pyr_vfe_dft_meV)),
            "dft_barrier_meV": round(max(pyr_vfe_dft_meV) - min(pyr_vfe_dft_meV), 3),
            "note": "Separate V_Fe canonical path (E_a=269 meV). Canonical MLIP "
                    "pyrite run is V_S, so no matched MLIP V_Fe band -> DFT-only.",
        })

    summary = {
        "review_question": "Q5: per-image MLIP-vs-DFT band correlation / failure typology",
        "metric_defs": {
            "pearson_R": "Pearson correlation, MLIP vs DFT over 9 images (None if a band is flat)",
            "rmse_meV": "RMSE of MLIP rel-energy vs DFT rel-energy, meV",
            "saddle_matches_dft": "argmax(MLIP band) == argmax(DFT band)",
            "barrier_ratio": "MLIP barrier / DFT barrier",
        },
        "cases": cases,
    }

    with open(P("tmp/q5_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- pretty print ----
    print("=" * 78)
    print("Q5  MLIP-vs-DFT per-image band correlation")
    print("=" * 78)
    for c in cases:
        print(f"\n### CASE: {c['case']}")
        print(f"    {c.get('note','')}")
        if "dft_band_meV" in c and "models" in c:
            hdr = f"{'img':>3} | {'DFT(meV)':>10}"
            for tag in c["models"]:
                hdr += f" | {tag[:14]:>14}"
            print("    " + hdr)
            print("    " + "-" * len(hdr))
            for i in range(c["n_images"]):
                row = f"{i:>3} | {c['dft_band_meV'][i]:>10.3f}"
                for tag in c["models"]:
                    row += f" | {c['models'][tag]['mlip_band_meV'][i]:>14.4f}"
                print("    " + row)
            print(f"    DFT  barrier = {c['dft_barrier_meV']:.3f} meV @ img{c['dft_saddle_index']}")
            for tag, m in c["models"].items():
                R = m["pearson_R"]
                Rs = "N/A(flat)" if R is None else f"{R:+.4f}"
                print(f"    {tag:>15}: barrier={m['mlip_barrier_meV']:.4f} meV "
                      f"@ img{m['mlip_saddle_index']} (saddle_match={m['saddle_matches_dft']}) "
                      f"| R={Rs} | RMSE={m['rmse_meV']:.3f} meV "
                      f"| ratio={m['barrier_ratio_mlip_over_dft']}")
                print(f"        --> {c['failure_classification'][tag]}")
        elif "mace_band_meV" in c:
            print(f"    MACE band (meV):   {c['mace_band_meV']}")
            print(f"    CHGNet band (meV): {c['chgnet_band_meV']}")
        elif "dft_band_meV" in c:
            print(f"    DFT band (meV): {c['dft_band_meV']}")
            print(f"    barrier = {c['dft_barrier_meV']:.3f} meV @ img{c['dft_saddle_index']}")
        if "failure_classification" in c and "models" not in c:
            for tag, lab in c["failure_classification"].items():
                print(f"    {tag}: {lab}")

    print("\nSaved: tmp/q5_summary.json")


if __name__ == "__main__":
    main()
