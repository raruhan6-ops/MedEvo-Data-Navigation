# Results (SLAKE, English)

Accuracy in %, differences in percentage points (pp).

## Table 1 — Base model (C0) zero-shot accuracy

| Split | Closed | Open | Open recall | Overall | N questions |
|---|---|---|---|---|---|
| Feedback (val) | 76.54 | 34.55 | 40.59 | 51.38 | 1053 |
| Test | 75.72 | 36.59 | 42.76 | 51.93 | 1061 |

## Table 2 — Capability gaps on the feedback set (C0), sorted by error rate

| Axis | Tag group | N | Error rate | Navigation quota (of 1500) |
|---|---|---|---|---|
| content_type | KG | 155 | 72.3 | 101 |
| content_type | Color | 31 | 67.7 | 95 |
| content_type | Plane | 62 | 67.7 | 95 |
| location | Neck | 40 | 57.5 | 80 |
| location | Brain_Tissue | 173 | 56.1 | 78 |
| content_type | Position | 183 | 55.7 | 78 |
| modality | X-Ray | 338 | 52.7 | 73 |
| content_type | Size | 59 | 52.5 | 73 |
| location | Pelvic Cavity | 42 | 52.4 | 73 |
| location | Lung | 386 | 51.0 | 71 |
| content_type | Quantity | 56 | 50.0 | 70 |
| content_type | other | 11 | 48.6 * | 68 |
| modality | MRI | 233 | 48.5 | 68 |
| modality | CT | 482 | 45.9 | 64 |
| location | other | 22 | 45.5 | 63 |
| location | Abdomen | 296 | 43.2 | 60 |
| content_type | Abnormality | 132 | 40.9 | 57 |
| location | Brain_Face | 40 | 40.0 | 56 |
| location | Chest_lung | 27 | 37.0 | 52 |
| content_type | Organ | 254 | 33.9 | 47 |
| location | Brain | 27 | 33.3 | 46 |
| content_type | Modality | 110 | 22.7 | 32 |

\* group below the minimum size; uses the overall feedback error rate.

## Table 3 — Main comparison on SLAKE test (equal budget B)

| Condition | Training examples | Closed | Open | Overall (mean ± SD) | Δ vs C0 |
|---|---|---|---|---|---|
| C0 Base | 0 | 75.72 | 36.59 | 51.93 | — |
| C1 Random | 1500 | 85.58 | 52.92 | 65.72 ± 0.33 (n=3) | +13.79 |
| C2 Navigation | 1500 | 86.62 | 55.30 | 67.58 ± 0.34 (n=3) | +15.65 |
| C3 All correct rollouts | 3065 | 86.78 | 55.97 | 68.05 | +16.12 |
| C4 Navigation, round 2 (α = 1) | 1500 | 85.58 | 55.50 | 67.30 | +15.37 |

**C2 − C1 (seed-averaged): +1.85 pp, 95% paired bootstrap CI [+0.79, +2.95]** (10,000 question resamples).

- seed 1: +1.98 pp [+0.47, +3.58]
- seed 2: +2.45 pp [+0.94, +3.96]
- seed 3: +1.13 pp [-0.47, +2.83]

### Robustness — lenient scoring

| Condition | Overall, strict | Overall, lenient | Lenient − strict |
|---|---|---|---|
| C0 Base | 51.93 | 56.46 | +4.52 |
| C1 Random | 65.72 ± 0.33 | 68.08 ± 0.52 | +2.36 |
| C2 Navigation | 67.58 ± 0.34 | 69.84 ± 0.34 | +2.26 |
| C3 All correct rollouts | 68.05 | 70.31 | +2.26 |
| C4 Navigation, round 2 (α = 1) | 67.30 | 69.56 | +2.26 |

C2 − C1 under lenient scoring: +1.76 pp, 95% CI [+0.72, +2.83].

Lenient scoring is a secondary check, not the primary metric. Beyond the strict normalisation it treats 'axial' as 'transverse', ignores a trailing plural 's', and drops the word 'plane' (e.g. 'axial' = 'Transverse Plane', 'Lungs' = 'Lung'). Training data and selection used strict scoring only.

Reference: Paper, Qwen3.5-9B on SLAKE: 76.80 (R0) → 78.23 (R3); not directly comparable.

## Table 4 — Per-group change on test (C1 vs C2)

| Tag group | N test | C0 | C1 | C2 | C2 − C1 | Targeted by navigation? |
|---|---|---|---|---|---|---|
| location:other | 10 | 30.0 | 46.7 | 60.0 | +13.3 | yes (4% vs 2%) |
| content_type:Organ | 253 | 66.8 | 79.4 | 83.8 | +4.3 | no (27% vs 31%) |
| location:Brain_Tissue | 166 | 49.4 | 59.6 | 63.5 | +3.8 | yes (14% vs 12%) |
| location:Neck | 35 | 45.7 | 57.1 | 61.0 | +3.8 | yes (4% vs 2%) |
| content_type:KG | 148 | 26.4 | 38.5 | 41.2 | +2.7 | yes (10% vs 6%) |
| modality:CT | 472 | 51.5 | 62.9 | 65.3 | +2.4 | yes (55% vs 49%) |
| content_type:Plane | 58 | 37.9 | 46.0 | 48.3 | +2.3 | yes (6% vs 3%) |
| location:Pelvic Cavity | 47 | 29.8 | 40.4 | 42.6 | +2.1 | yes (8% vs 4%) |
| content_type:Quantity | 52 | 48.1 | 71.2 | 73.1 | +1.9 | yes (7% vs 4%) |
| location:Abdomen | 313 | 54.0 | 66.2 | 68.2 | +1.9 | no (24% vs 33%) |
| modality:MRI | 228 | 56.1 | 66.8 | 68.7 | +1.9 | no (22% vs 24%) |
| content_type:Abnormality | 150 | 54.7 | 64.9 | 66.0 | +1.1 | no (13% vs 16%) |
| modality:X-Ray | 361 | 49.9 | 68.7 | 69.8 | +1.1 | no (24% vs 27%) |
| content_type:Position | 186 | 39.2 | 47.5 | 48.6 | +1.1 | no (11% vs 13%) |
| content_type:Size | 65 | 47.7 | 83.1 | 84.1 | +1.0 | no (8% vs 8%) |
| location:Lung | 419 | 52.7 | 70.3 | 71.2 | +0.9 | no (29% vs 34%) |
| location:Brain | 42 | 73.8 | 77.8 | 78.6 | +0.8 | no (6% vs 6%) |
| content_type:Modality | 108 | 88.0 | 97.5 | 97.5 | +0.0 | no (12% vs 16%) |
| content_type:other | 7 | 0.0 | 0.0 | 0.0 | +0.0 | no (0% vs 0%) |
| location:Brain_Face | 29 | 51.7 | 69.0 | 67.8 | -1.1 | yes (6% vs 3%) |
| content_type:Color | 34 | 44.1 | 90.2 | 84.3 | -5.9 | yes (6% vs 3%) |

Targeted = the group's share of navigation examples exceeds its share of usable pool questions (shown as nav share vs pool share). Small groups are noisy.

## Table 5 — Data statistics per condition

| Selection | Questions sampled | With ≥1 correct rollout | Accepted examples | Correct rollouts (%) | Share from weakest 3 groups (%) |
|---|---|---|---|---|---|
| all | 4919 | 3065 | 3065 | 45.3 | 12.2 |
| nav | 3354 | 1500 | 1500 | 23.2 | 22.2 |
| nav_r2 | 3354 | 1500 | 1500 | 23.6 | 23.1 |
| random_s1 | 2399 | 1500 | 1500 | 45.1 | 12.5 |
| random_s2 | 2399 | 1500 | 1500 | 45.8 | 12.5 |
| random_s3 | 2413 | 1500 | 1500 | 45.2 | 12.8 |

Weakest 3 groups (C0 feedback): content_type:KG, content_type:Color, content_type:Plane. Questions sampled counts every pool question the selector walked, including those skipped for having no correct rollout.
