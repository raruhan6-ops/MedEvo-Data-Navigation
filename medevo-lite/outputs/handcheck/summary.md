# Hand-check of parser and scorer (SLAKE test)

| Condition | Checked | Parser right | Scorer agrees with person | Strict too harsh | Strict too lenient |
|---|---|---|---|---|---|
| C0 | 50 | 50 (100%) | 47 (94%) | 3 | 0 |
| C1_s1 | 50 | 50 (100%) | 46 (92%) | 4 | 0 |
| C2_s1 | 50 | 50 (100%) | 47 (94%) | 3 | 0 |
| C3 | 50 | 50 (100%) | 47 (94%) | 3 | 0 |

Strict too harsh = marked wrong, a person would accept it. Strict too lenient = marked right, a person would not.

## Disagreements and parser errors

- C0 q12722: gold `Transverse Plane`, pred `axial`, strict 0, person y (axial = transverse plane)
- C0 q12219: gold `Transverse  Plane`, pred `axial`, strict 0, person y (axial = transverse plane)
- C0 q12589: gold `Transverse Plane`, pred `axial`, strict 0, person y (axial = transverse plane)
- C1_s1 q12722: gold `Transverse Plane`, pred `Axial`, strict 0, person y (axial = transverse plane)
- C1_s1 q12219: gold `Transverse  Plane`, pred `Axial`, strict 0, person y (axial = transverse plane)
- C1_s1 q12433: gold `Left Lung`, pred `Left`, strict 0, person y (side only; borderline)
- C1_s1 q12589: gold `Transverse Plane`, pred `Axial`, strict 0, person y (axial = transverse plane)
- C2_s1 q12722: gold `Transverse Plane`, pred `axial`, strict 0, person y (axial = transverse plane)
- C2_s1 q12219: gold `Transverse  Plane`, pred `Axial`, strict 0, person y (axial = transverse plane)
- C2_s1 q12589: gold `Transverse Plane`, pred `Axial`, strict 0, person y (axial = transverse plane)
- C3 q12722: gold `Transverse Plane`, pred `Axial`, strict 0, person y (axial = transverse plane)
- C3 q12219: gold `Transverse  Plane`, pred `Axial`, strict 0, person y (axial = transverse plane)
- C3 q12589: gold `Transverse Plane`, pred `Axial`, strict 0, person y (axial = transverse plane)
