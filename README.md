# scnmssm-boosted-tau-bdt

Reproduction scripts for the classification results in arXiv:2609.29620 (boosted multi-tau signature of a light pseudoscalar in the scNMSSM, BDT against Z→ττ).

## Environment

Python 3.10.12, NumPy 2.2.6, scikit-learn 1.7.2, Matplotlib 3.10. All seeds are fixed (42). Scripts take no arguments and must be run from the folder that holds the `.npy` arrays.

## Scripts

| File | What it does |
|---|---|
| `train_2level.py` | Gradient-boosted classifier at jet level (9 features) and event level (32 features); prints AUC and signal efficiency at fixed background rates. |
| `domain_shift.py` | Cross-slice test: train on Z pT(ττ)>100 GeV, test on pT(ττ)>50 GeV, and the reverse (Table 5). |
| `robustness_check.py` | Event-level feature-set ablation (32 / 28 / 7 / 2 inputs). **Not re-run in the demo session below.** |
| `measure_signal_overlap.py` | Counts signal events shared between the training set of one slice model and the test set of the other. |
| `domain_shift_28.py` | Cross-slice test repeated without HT and the three fat-jet pT inputs (28 inputs). |

## Results re-run on 26 October 2026

| Quantity | Re-run | Paper |
|---|---|---|
| Jet-level AUC | 0.7568 | 0.757 |
| Event-level AUC | 0.9405 | 0.941 |
| Cross-slice AUC, 100→100 / 100→50 / 50→50 / 50→100 | 0.9360 / 0.7268 / 0.9095 / 0.8670 | 0.936 / 0.727 / 0.910 / 0.867 |

Cross-slice test with 28 inputs (no HT, no j{k}_pT): 0.9315 / 0.7952 / 0.9064 / 0.8723. The 100→50 drop falls from 0.209 to 0.136 but does not vanish. MET is still among the 28 inputs, so nothing is claimed about it.

## Known limitations

- Each cross-slice number comes from one 70/30 split and one seed, with 6,000 events per class in each test set, and no uncertainty is given. Cross-validation is not done yet.
- The signal events of the two slices are drawn independently from one pool, so about 14% of the signal events in each shifted test set were also in the training set (measured: 843/6,000 and 829/6,000). This can only make the shifted AUC look better.
- Why the AUC drops is inferred from sample means, not shown by a feature-level measurement.
- The jet-level split may place jets of the same event in both training and test sets, so the jet-level AUC may be slightly optimistic.

## Change to an existing file

In `train_2level.py` only the figure path on line 84 was changed (absolute path replaced by `ROC_2level.png`). The original file had sha256 `fb2e9ada3f8e02b686ae7c193e63ef15020691c5e0eb4b354a9625b45fed81e2`.

## Demo session

`demo_transcript.txt` is an unedited export of one Claude Code session (Opus 5.5, manual-approval mode) in which I re-ran two of these scripts and checked the claims of my technical record. The prompts were prepared in advance with the help of another Claude conversation, and my two closing replies were drafted the same way; the decisions in them (accept the referee point on the momentum variables, defer the MET test and cross-validation) were mine. The "expected AUC" in the fourth prompt was not blind, because the record with Table 5 was already in context. The session found and corrected two errors in my record: the test-set size (6,000 events per class, not 3,000) and an over-broad sentence about the momentum variables.

## Data

The input arrays (`evtlevel_*.npy`, `jetlevel_*.npy`) are not stored here; they are being deposited on Zenodo and the link will be added to this page.

## Licence

Code: MIT.
