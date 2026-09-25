"""Per-image MLIP-vs-DFT statistics for all five bands, with marcasite image 1 excluded.

DFT bands come from the deposit (the published artefact, so the paper and the deposit cannot
drift apart); MLIP bands come from the R2.2 run. Writes a JSON so the manuscript tables are
generated from measured numbers rather than retyped.
"""
import json
import math
import os
from pathlib import Path

from ase.io import read

# Paths are taken from the environment so the script runs from a clone without editing:
#   MLIP_RUN  directory holding the R2.2 run output (band_*.json, grace_*.json)
#   DEPOSIT   this repository's data/structures directory
#   MLIP_STATS_OUTPUT output JSON (default tmp/per_image_stats.json in the clone)
REPO = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("MLIP_RUN", REPO / "mlip/r22_2026-09"))
DEP = Path(os.environ.get("DEPOSIT", REPO / "data/structures"))

# band key in the R2.2 run  ->  deposited DFT band
DEPOSIT = {
    "mackinawite": "mackinawite_VFe_band.extxyz",
    "pyrite_VS2": "pyrite_VS2_band.extxyz",
    "marcasite": "marcasite_VFe_band.extxyz",
    "greigite_channel": "greigite_VFe_channel_band.extxyz",
    "greigite_cation": "greigite_VFe_cation_band.extxyz",
}

# images whose stored DFT energy the deposit itself flags as unreliable
EXCLUDE = {"marcasite": [1]}


def dft_band(fn):
    frames = read(DEP / fn, index=":")
    es = []
    for at in frames:
        e = None
        for k in ("energy", "free_energy", "meta_energy_eV", "meta_energy"):
            if k in at.info:
                e = at.info[k]
                break
        if e is None:
            e = at.get_potential_energy()
        es.append(e)
    return [(e - es[0]) * 1000.0 for e in es]


def mlip_bands(band):
    out = {}
    for fn in (f"band_{band}.json", f"grace_{band}.json"):
        p = RES / fn
        if not p.exists():
            continue
        for key, val in json.loads(p.read_text()).items():
            if isinstance(val, dict) and "profile_meV" in val:
                out[key] = val["profile_meV"]
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def rmse(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def mirror(p):
    n = len(p)
    return max(abs(p[i] - p[n - 1 - i]) for i in range(n))


report = {}
for band, fn in DEPOSIT.items():
    dft = dft_band(fn)
    models = mlip_bands(band)
    if len(models) != 9:
        raise SystemExit(f"Expected nine model profiles for {band}, found {len(models)} in {RES}")
    drop = EXCLUDE.get(band, [])
    keep = [i for i in range(len(dft)) if i not in drop]
    ea_dft = max(dft) - dft[0]
    sad_dft = dft.index(max(dft))

    print(f"\n{'=' * 96}\n{band}   E_a(DFT) = {ea_dft:.1f} meV   saddle img {sad_dft}   "
          f"mirror {mirror(dft):.2f} meV   E_rxn {dft[-1]:.1f} meV")
    print("  DFT   :", " ".join(f"{v:7.1f}" for v in dft))
    if drop:
        print(f"  ! image(s) {drop} excluded from per-image statistics (deposit flags the energy)")

    hdr = f"  {'model':<34} {'E_a':>7} {'diff':>7} {'R':>6} {'RMSE':>7} {'saddle':>6} {'mirror':>8}"
    if drop:
        hdr += f" | {'R*':>6} {'RMSE*':>7}"
    print(hdr)

    rows = {}
    for m in sorted(models):
        p = models[m]
        if not isinstance(p, list) or len(p) != len(dft):
            raise SystemExit(f"Invalid profile length for {band}/{m}: expected {len(dft)} images")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p):
            raise SystemExit(f"Non-finite or non-numeric profile for {band}/{m}")
        ea = max(p) - p[0]
        r = pearson(dft, p)
        rm = rmse(dft, p)
        ok = "yes" if p.index(max(p)) == sad_dft else "NO"
        row = {"E_a": round(ea, 1), "diff": round(ea - ea_dft, 1),
               "R": None if r is None else round(r, 3), "rmse": round(rm, 1),
               "saddle_match": ok == "yes", "mirror": round(mirror(p), 3)}
        line = (f"  {m:<34} {ea:7.1f} {ea - ea_dft:7.1f} "
                f"{(r if r is not None else float('nan')):6.2f} {rm:7.1f} {ok:>6} {mirror(p):8.3f}")
        if drop:
            d2 = [dft[i] for i in keep]
            p2 = [p[i] for i in keep]
            r2 = pearson(d2, p2)
            row["R_excl"] = None if r2 is None else round(r2, 3)
            row["rmse_excl"] = round(rmse(d2, p2), 1)
            line += f" | {(r2 if r2 is not None else float('nan')):6.2f} {rmse(d2, p2):7.1f}"
        print(line)
        rows[m] = row

    report[band] = {"dft_profile_meV": [round(v, 2) for v in dft],
                    "E_a_dft_meV": round(ea_dft, 2),
                    "saddle_image": sad_dft,
                    "excluded_images": drop,
                    "models": rows}

out = Path(os.environ.get("MLIP_STATS_OUTPUT", REPO / "tmp/per_image_stats.json"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(f"\nwritten: {out}")
