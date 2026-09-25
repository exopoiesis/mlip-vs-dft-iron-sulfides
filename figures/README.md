# Figure generators

The numbered filenames are retained for continuity; the current manuscript uses:

| Generator | Current location |
|---|---|
| `fig3_cross_mineral_benchmark.py` | Main Figure 1 |
| `fig4_zpe_cross_family.py` | Main Figure 2 |
| `fig1_pyrite_anchor.py` | SI Figure R1 |
| `fig2_mackinawite_vfe.py` | SI Figure R2 |
| `fig5_barrier_landscape.py` | SI Figure R3 |

Install the core requirements in a virtual environment, then run a generator from any working directory, for example `python figures/fig1_pyrite_anchor.py`. It writes PDF, PNG and SVG beside the script. No DFT or MLIP calculation is launched. JSON inputs are in `data/figure_inputs/`; their original bytes and SHA-256 hashes are recorded in `data/figure_inputs/provenance.json`. The comparison figures also contain explicitly tabulated manuscript/literature values; not every plotted value is recomputed by the plotting script.

`figS_mlip_dft_band_correlation.py` is a historical diagnostic, excluded from the current figures. It contrasts the valid DFT pyrite band with degenerate-endpoint MLIP profiles. Those MLIP ranges are not migration barriers; the old failure-mode interpretation is withdrawn (SI O.2).

`withdrawn_pentlandite_endpoints.py` reports why the former figure was withdrawn. Its damaged, non-executable historical source is preserved verbatim as `withdrawn_pentlandite_endpoints.py.txt`; it is not repaired into an apparently valid calculation. The starting structure was incorrect, as documented in `../mlip/PENTLANDITE_WITHDRAWN.md`.
