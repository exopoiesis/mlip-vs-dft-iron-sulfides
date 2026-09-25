# Deposited plotting inputs

Five JSON files are byte-exact copies of the local calculation summaries previously required from an external workspace by the pyrite and mackinawite plotting scripts. `provenance.json` records their source paths, hashes and sizes. These paths identify the historical origin; execution uses the adjacent deposited copies.

The original summaries retain historical labels and interpretation fields. Current scientific interpretation is defined in the manuscript and SI, especially the distinct small-cell/production pyrite compositions and the withdrawn legacy V_S pathways. The plotting scripts use the energy arrays and numeric diagnostics; copying a source summary does not reinstate its old narrative claims.

`pyrite_legacy_profiles.json` extracts the pyrite case from the historical `tmp/q5_summary.json`. Its DFT and MLIP numbers are unchanged. Its top-level status explicitly states that the MLIP endpoints are degenerate and the profiles are not migration-barrier predictions. It supports only the historical diagnostic plot.
