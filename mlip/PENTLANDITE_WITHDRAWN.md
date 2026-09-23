# Pentlandite: withdrawn (2026-09-23)

Everything in this deposit that concerns pentlandite was computed on a cell that reproduces
the **composition** of pentlandite but not its **structure**. The files are kept, because
releases v1.0-v1.3 contain them and may have been cited; they are flagged rather than deleted.

## What was wrong

| site | pentlandite (published) | the cell we used |
|---|---|---|
| octahedral metal | Wyckoff **4b** (1/2, 1/2, 1/2) | (0, 0, 0), i.e. 4a |
| tetrahedral metal | **32f**, x = 0.1261 | 32f, x = 0.356 |
| sulfur | **8c** (1/4, 1/4, 1/4) **and 24e** (0.2629, 0, 0) | a single 32f orbit, x = 0.118 |
| lattice parameter | 9.928 A experimental / 9.948 A DFT Fe9S8 | 10.07 A |

Both descriptions give 36 metals and 32 sulfurs per conventional cell, hence Fe9S8, so every
composition check passed. The coordination does not agree: measured on the cell we used, the
tetrahedral metal has **3** S at 2.425 A and the octahedral metal **8** S at 2.058 A, against
the 4 and 6 that pentlandite requires. In the DFT-relaxed endpoints deposited here, one iron
has no sulfur within 2.9 A at all.

## What is withdrawn

* the [Fe4S4] "cubane collapse" foundation-MLIP failure mode. The "3 + 3" signature is the
  starting geometry of that cell - the long Fe-S distance is 3.574 A before any potential is
  applied, and MACE-MP-0 moves it by 0.001 A. The comparison baseline used in the SI, an
  idealised Td cubane with four equal 2.10-2.16 A bonds, does not exist in that structure.
* the V_S + H pathway breakdown attributed to pentlandite, which rested on all of its sulfur
  being symmetry-equivalent - a property of the incorrect cell.
* the MD "cage-rattling" consistency check and the 1.43 eV MLIP barrier derived from it.

The motif was also misnamed: pentlandite contains an **[M8S6]** cluster - eight metals on the
corners of a cube, twelve M-M edges of 2.509 A, six sulfurs capping the faces - not the
[Fe4S4] cubane of ferredoxins, in which metal and sulfur alternate on the corners.

## What replaces it

`pentlandite_structure_verified.py` builds the mineral from the published Wyckoff positions and
**refuses to return a cell whose metal coordination is not {4: 8N, 6: N}**. A composition check
cannot catch this class of error; a coordination check catches it in one line.

Relaxing the correct 136-atom cell with MACE-MP-0 large (same checkpoint as the paper) converges
in a single optimiser step: metal-metal 2.5089 -> 2.5098 A, every coordination number preserved.
A vacancy at either metal site also leaves all coordination numbers unchanged. There is no
cubane failure mode on the correct structure, and no contrast between the two vacancy sites.

## What is unaffected

Pyrite, marcasite, mackinawite and greigite carry every DFT barrier in the paper. Their deposited
structures were re-checked against published crystallography: pyrite 31 Fe at coordination 6,
2.257 A; marcasite 31 Fe at 6, 2.233 A; mackinawite 35 Fe at 4, 2.171 A; greigite 8 Fe at 4
(2.19-2.21 A) plus 15 at 6 (2.40-2.44 A), the inverse thiospinel with one octahedral vacancy.
No barrier, no model benchmark and no fine-tuning result changes.
