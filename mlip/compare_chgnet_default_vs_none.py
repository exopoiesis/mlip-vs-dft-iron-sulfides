#!/usr/bin/env python3
"""Compare CHGNet mack V_Fe scan: --magmom-mode default vs none.
Quick statistics — does S-H % shift as predicted?
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results/mlip_active_learning/gomer_s132_vfe_scan/chgnet/chgnet_mack_vfe_scan.json"
NONE_JSON = ROOT / "results/mlip_active_learning/gomer_s132_mack_vfe_chgnet_none/chgnet_mack_vfe_scan_none.json"

def categorize(j):
    rows = []
    n_total = j['n_valid']
    for c in j['clusters']:
        rep = c['rep_h_nearest']
        rows.append({'cid': c['cluster_id'], 'size': c['size'],
                     'pct': 100*c['size']/n_total, 'sym': rep['symbol'],
                     'idx': rep['index'], 'd': rep['distance_A']})
    s_h = [r for r in rows if r['sym']=='S' and 1.20<=r['d']<=1.55]
    fe_h = [r for r in rows if r['sym']=='Fe' and 1.50<=r['d']<=2.10]
    anom = [r for r in rows if r not in s_h and r not in fe_h]
    return {
        'n_valid': n_total,
        'n_clusters': len(rows),
        's_h_size': sum(r['size'] for r in s_h),
        's_h_clusters': len(s_h),
        's_h_distinct_10pct': sum(1 for r in s_h if r['size']>=0.10*n_total),
        'fe_h_size': sum(r['size'] for r in fe_h),
        'fe_h_clusters': len(fe_h),
        'anom_size': sum(r['size'] for r in anom),
        'top5': sorted(rows, key=lambda x: -x['size'])[:5],
    }

print("="*80)
print("CHGNet mack V_Fe — default (AFM-layer) vs none (zero magmom)")
print("="*80)

if not DEFAULT_JSON.exists():
    print(f"DEFAULT not found: {DEFAULT_JSON}")
    exit(1)
if not NONE_JSON.exists():
    print(f"NONE not found: {NONE_JSON}")
    exit(1)

j_def = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
j_none = json.loads(NONE_JSON.read_text(encoding="utf-8"))
s_def = categorize(j_def)
s_none = categorize(j_none)

print(f"\n{'Metric':<35} {'default (AFM)':>18} {'none (zero)':>18}")
print("-" * 75)
print(f"{'V_Fe atom index':<35} {j_def.get('V_Fe_index','?'):>18} {j_none.get('V_Fe_index','?'):>18}")
print(f"{'S neighbours':<35} {str(j_def.get('S_neighbours',[]))[:18]:>18} {str(j_none.get('S_neighbours',[]))[:18]:>18}")
print(f"{'E_pristine (eV)':<35} {j_def['E_pristine_eV']:>18.4f} {j_none['E_pristine_eV']:>18.4f}")
print(f"{'n_valid':<35} {s_def['n_valid']:>18} {s_none['n_valid']:>18}")
print(f"{'n_clusters total':<35} {s_def['n_clusters']:>18} {s_none['n_clusters']:>18}")
print(f"{'S-H members (1.2-1.55 A)':<35} {s_def['s_h_size']:>18} {s_none['s_h_size']:>18}")
print(f"{'S-H % of valid':<35} {100*s_def['s_h_size']/s_def['n_valid']:>17.1f}% {100*s_none['s_h_size']/s_none['n_valid']:>17.1f}%")
print(f"{'S-H clusters':<35} {s_def['s_h_clusters']:>18} {s_none['s_h_clusters']:>18}")
print(f"{'S-H distinct >=10%':<35} {s_def['s_h_distinct_10pct']:>18} {s_none['s_h_distinct_10pct']:>18}")
print(f"{'Fe-H members (1.5-2.10 A)':<35} {s_def['fe_h_size']:>18} {s_none['fe_h_size']:>18}")
print(f"{'Fe-H % of valid':<35} {100*s_def['fe_h_size']/s_def['n_valid']:>17.1f}% {100*s_none['fe_h_size']/s_none['n_valid']:>17.1f}%")
print(f"{'Anomaly members':<35} {s_def['anom_size']:>18} {s_none['anom_size']:>18}")

print(f"\nTop 5 clusters (default):")
for r in s_def['top5']:
    print(f"  cid {r['cid']:>3} size={r['size']:>2} ({r['pct']:>4.1f}%) {r['sym']} idx{r['idx']} d={r['d']:.3f} A")
print(f"Top 5 clusters (none):")
for r in s_none['top5']:
    print(f"  cid {r['cid']:>3} size={r['size']:>2} ({r['pct']:>4.1f}%) {r['sym']} idx{r['idx']} d={r['d']:.3f} A")

# Verdict
def verdict(s):
    if s['s_h_distinct_10pct']>=2 and s['s_h_size']>=0.30*s['n_valid']:
        return "GO"
    elif s['s_h_size']>=0.10*s['n_valid'] and s['s_h_distinct_10pct']>=1:
        return "AMBIGUOUS"
    else:
        return "NO-GO"

print(f"\n{'Verdict':<35} {verdict(s_def):>18} {verdict(s_none):>18}")

# Direction shift
shift = 100*s_none['s_h_size']/s_none['n_valid'] - 100*s_def['s_h_size']/s_def['n_valid']
print(f"\nS-H % shift (none − default): {shift:+.1f} percentage points")
if shift > 5:
    print("→ chemist hypothesis correct (none stabilizes S-H, AFM frustrated Fe basin)")
elif shift < -5:
    print("→ physicist hypothesis correct (none destabilizes Fe d-band, more Fe-collapse)")
else:
    print("→ no strong shift, both runs effectively similar")
