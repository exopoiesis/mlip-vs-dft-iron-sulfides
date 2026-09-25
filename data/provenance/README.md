# Primary method provenance

The two image-4 Quantum ESPRESSO input files for the corrected greigite channel and cation paths record `ecutwfc = 80 Ry` and `ecutrho = 320 Ry`. The channel output independently confirms those settings and records the ONCV pseudopotential valences and MD5 checksums. The generic `Fe.upf`, `S.upf` and `H.upf` filenames alone do not identify a vendor release.

These files are byte-exact copies of the retained production artifacts. Their source paths, sizes and SHA-256 hashes are in `greigite_method_provenance.json`. The `.in`/`.out` extensions replace the historical `.pwi`/`.pwo` names only to distinguish these small intentionally deposited records from regenerable runtime output. No file contents were rewritten.

This is evidence for the reported method settings, not a complete archive of all intermediate SCF files or the pseudopotential files themselves. Execution would require adapting the absolute runtime paths and supplying the matching pseudopotentials. No new DFT calculation was run for this correction.
