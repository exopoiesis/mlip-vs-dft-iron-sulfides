"""GRACE over every band. Writes one json per band with the model name in the filename, so a
second pass cannot silently overwrite a first -- the mistake that cost the torch-side results
their json earlier tonight (recovered from logs, but only because the logs were kept)."""
import json
from pathlib import Path
import numpy as np
from ase.io import read

BANDS = {
    "mackinawite":      ("/data/bands/mackinawite", 42.9),
    "pyrite_VS2":       ("/data/bands/pyrite_VS2", 94.6),
    "marcasite":        ("/data/bands/marcasite", 208.2),
    "greigite_channel": ("/data/bands/greigite_channel", 236.0),
    "greigite_cation":  ("/data/bands/greigite_cation", 566.8),
}
MODELS = ["GRACE-1L-OMAT", "GRACE-2L-OMAT", "GRACE-2L-OAM"]

from tensorpotential.calculator import grace_fm  # noqa: E402

calcs = {}
for name in MODELS:
    try:
        calcs[name] = grace_fm(name)
        print(f"loaded {name}")
    except Exception as e:
        print(f"load FAILED {name}: {type(e).__name__}: {str(e)[:120]}")

for tag, (path, dft) in BANDS.items():
    p = Path(path)
    files = sorted(p.glob("final_0*.xyz"))
    imgs = [read(str(f)) for f in files] if files else read(str(next(p.glob("*.extxyz"))), index=":")
    print(f"\nband: {len(imgs)} images from {path}, {imgs[0].get_chemical_formula()}")
    out = {}
    for name, calc in calcs.items():
        print(f"=== {name} ===")
        try:
            E = []
            for im in imgs:
                a = im.copy(); a.calc = calc
                E.append(a.get_potential_energy())
            rel = (np.array(E) - E[0]) * 1000
        except Exception as e:
            print(f"  evaluation FAILED: {type(e).__name__}: {str(e)[:120]}")
            continue
        ea = float(rel.max())
        half = len(rel) // 2 + 1
        out[name] = dict(profile_meV=[round(float(x), 2) for x in rel],
                         E_a_meV=round(ea, 2), diff_vs_dft_meV=round(ea - dft, 2),
                         endpoint_drift_meV=round(float(rel[-1]), 3),
                         monotone_to_saddle=bool(np.all(np.diff(rel[:half]) > 0)),
                         any_below_endpoint=bool(rel.min() < -1.0))
        print(f"  E_a = {ea:7.1f} meV   DFT {dft}   diff {ea-dft:+.1f}")
        print(f"  drift {out[name]['endpoint_drift_meV']:+.2f}  "
              f"monotone {out[name]['monotone_to_saddle']}  "
              f"below-endpoint {out[name]['any_below_endpoint']}")
    Path(f"/data/grace_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote /data/grace_{tag}.json")
