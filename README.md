# Training Regime, Not Backbone: missing-view robustness in multimodal skin cancer diagnosis

Code, fold assignments and out-of-fold predictions for the paper:

> **Training Regime, Not Backbone: A Controlled Evaluation of Missing-View Robustness in Multimodal Skin Cancer Diagnosis.**
> Al-Kubaisi M, et al. Submitted to *Computers in Biology and Medicine*, 2026.

The study trains clinical–dermoscopic fusion networks on **MILK10k** (5,240 lesions, 11 classes) in three view states: paired, clinical-only and dermoscopic-only. It compares:

- joint view-state training;
- modality dropout;
- mean imputation;
- a paired-only model;
- single-view specialists.

The comparisons use two backbones (ConvNeXt-Tiny and PanDerm-Base) and lesion-level 5-fold cross-validation. Models are then evaluated at a 95% malignant-sensitivity operating point, for dermoscopy referral and by skin tone. External testing uses Derm7pt, PAD-UFES-20 and the MILK10k blind benchmark.

## Repository layout

| Folder | Content |
|---|---|
| `notebooks/` | Kaggle notebooks used for training and evaluation: round 1 (ConvNeXt-T), round 2 (PanDerm-B), round 3A (modality dropout, mean imputation, extra seeds) and round 3B (ConvNeXt-T with the matched recipe). `pipeline_source_round3.py` is the plain-Python source of the round-3 notebooks. |
| `data/folds.csv` | Lesion-level 5-fold assignment (stratified, seed 42), identical for every model. |
| `data/lesion_tone.csv` | Skin-tone group per lesion (darker 0-2, medium 3, lighter 4-5), derived from MILK10k metadata, in `lesion_id` order. |
| `predictions_csv/` | Out-of-fold predictions for all 14 models: temperature-scaled class probabilities for each view state, the 95%-sensitivity threshold fitted on inner validation data, and the referral score where available. |
| `results/` | Raw outputs of each run: `oof.npz`, metrics, temperatures, training logs, skin-tone tables and external-validation predictions. `res1` is ConvNeXt round 1, `res2` is PanDerm round 2, `res3A` and `res3B` are round 3. |
| `benchmark/` | Probability files submitted to the MILK10k blind benchmark. |
| `analysis/` | Scripts that reproduce the paper's tables from `results/`: bootstrap comparisons, equivalence tests, referral at matched sensitivity, skin tone (including GEE pooling), and external validation. |
| `figures/` | `make_figs.py` and Figures 1-3. |

## Model names in `predictions_csv/`

| File | Model in the paper |
|---|---|
| `convnext_original_kd` | ConvNeXt-T joint model, original recipe (with self-distillation) |
| `convnext_no_kd` | ConvNeXt-T joint model without distillation |
| `convnext_matched_recipe` | ConvNeXt-T joint model, PanDerm schedule |
| `panderm_joint_seed1/2/3` | PanDerm-B joint model, three training seeds |
| `panderm_modality_dropout` | PanDerm-B, modality dropout (P = 0.5 / 0.25 / 0.25) |
| `panderm_mean_imputation` | PanDerm-B paired-only model with mean-embedding imputation |
| `*_paired_only`, `*_clinical_specialist`, `*_dermoscopic_specialist` | Baselines |

Malignant classes are AKIEC, BCC, MAL_OTH, MEL and SCCKA. The malignant probability is the sum of their probabilities.

## Reproducing the analyses

```bash
pip install -r requirements.txt
# run from the repository root
python analysis/round3/s3.py           # paired-bootstrap differences and equivalence (Tables 5, S1)
python analysis/round3/matched_ref.py  # referral at matched sensitivity (Table 7)
python analysis/round3/tone_all.py     # skin-tone odds ratios (Tables 8, S3)
python analysis/round3/ext_ci.py       # external validation with CIs (Table 9)
python analysis/round3/gee.py          # pooled skin-tone GEE (needs MILK10k metadata, see below)
```

`gee.py` and some round-1/2 scripts also need the official MILK10k files, which are not redistributed here. Download `MILK10k_Training_Metadata.csv` and `MILK10k_Training_GroundTruth.csv` from the ISIC Archive and place them in `data/`.

## Re-training

1. Open a notebook in `notebooks/` on Kaggle (GPU T4 x2, internet on).
2. Attach the datasets below, and run all cells.
3. Settings live in the CONFIG cell. In the round-3 notebooks, `RUN = "A"` or `"B"` selects the experiment set.

## Data (not included)

| Dataset | Source | Licence |
|---|---|---|
| MILK10k | ISIC Archive, doi:10.34970/648456 | CC-BY-NC |
| Derm7pt | https://derm.cs.sfu.ca | CC BY-NC-ND 4.0 |
| PAD-UFES-20 | Mendeley Data, doi:10.17632/zr7vgbcyr2.1 | CC BY 4.0 |
| PanDerm-Base weights | from the PanDerm authors (Yan et al., *Nat Med* 2025) | see the authors' terms |

Images and pretrained weights are not redistributed. Predictions and derived labels in this repository are shared for non-commercial research use, in line with the MILK10k licence.

## Licence

Code is released under the MIT licence (see `LICENSE`). Predictions and derived data follow the licences of the source datasets.

## Citation

Please cite the paper once published. Until then, cite this repository through its Zenodo DOI.
