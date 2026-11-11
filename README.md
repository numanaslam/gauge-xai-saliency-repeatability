# GAUGE-XAI

Run every script from the `code` folder. All outputs go to `../results/` (next to `code/`), so the code
folder holds code only:

```
paper/
  code/       scripts + gauge_lib.py, gauge_model.py
  results/
    audit/          00-03: manifests, lesion stats, QC montages, positive control
    lidc_patches/   04: nodules.csv, patches.npy (~8 GB), patches_qc.png
    runs/seed<N>/   05: best.pt, history.csv, summary.json, val_predictions.csv (+ split.csv)
    saliency/       06: measurands.csv, maps/, masks/
    analysis/       07: repeatability.csv, conditions.csv, seeds.csv, stability.csv, summary.txt, figures
    experiments/    Diagnostic training runs (2 mm, no prior, LR sweep, 64 mm FOV)
```

## Setup

```bash
pip install "pydicom>=2.4" numpy pandas matplotlib scipy statsmodels idc-index
# RTX 5070 (Blackwell, sm_120) needs a CUDA 12.8+ PyTorch build:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Experiment 0: data audit

```bash
python 00_lidc_check.py --root D:\data\lidc_idri                 # LIDC headers -> results/audit/lidc_series.csv
python 01_rider_audit.py --root D:\data\rider_lung_ct            # RIDER grid (or --from-manifest ..\results\audit\series_manifest.csv)
python 02_rider_qc.py                                            # CT+SEG load, lesion stats, QC montages, qc_ok
python 03_positive_control.py                                    # volume / diameter / HU repeatability
```

Done when every montage in `results/audit/qc/` shows the contour on the tumour, `qc_problems.txt` lists
only the rejected single-frame SEGs, and the positive control looks sensible .

## Experiment 1: model

```bash
python download_lidc.py --download-dir D:\data                   # only if the LIDC download is partial
python 04_lidc_patches.py --xml-root D:\data\LIDC-XML-only       # -> results/lidc_patches
python 05_train.py --seed 0                                      # -> results/runs/seed0 (~18 min)
python 05_train.py --seed 1
python 05_train.py --seed 2
```
 
## Experiment 2: saliency on RIDER

```bash
python 06_rider_saliency.py --patients RIDER-1129164940          # quick test: one patient (12 cells)
python 06_rider_saliency.py                                      # all usable cells x all seeds (resumable)
```
 
## Experiment 3: gauge statistics and figures

```bash
python 07_gauge_analysis.py                                      # -> results/analysis
``` 

## Robustness analyses

```bash
python 06_rider_saliency.py --gradcam-layer3     # adds Grad-CAM at layer3 (8 mm cells) + any new seeds only
python 07_gauge_analysis.py
python 10_robustness.py                          # -> results/analysis/robustness: RIDER sensitivity, LAF_adj,
                                                 #    shifted/mismatched-SEG controls, patient-cluster seed CIs,
                                                 #    mixed-model variance components (needs statsmodels)
```

## Centre-prior ablation (64 mm field of view, with vs. without the radial channel)

Both arms use 64^3 inputs (ROI 30 mm, controls shift 12.5 mm); 06, 07 and 10 pick the grid up from the models.
```bash
foreach ($s in 0,1,2) { python 05_train.py --input-size 64 --no-centre-prior --patches D:\paper8\results\lidc_patches_v2 --out D:\paper8\results\ablation\runs_noprior64 --seed $s }
foreach ($s in 0,1,2) { python 05_train.py --input-size 64 --patches D:\paper8\results\lidc_patches_v2 --out D:\paper8\results\ablation\runs_prior64 --seed $s }
foreach ($a in "noprior64","prior64") {
  python 06_rider_saliency.py --runs D:\paper8\results\ablation\runs_$a --out D:\paper8\results\ablation\saliency_$a
  python 07_gauge_analysis.py --saliency D:\paper8\results\ablation\saliency_$a --out D:\paper8\results\ablation\analysis_$a --no-lesion-figure
  python 10_robustness.py --saliency D:\paper8\results\ablation\saliency_$a --out D:\paper8\results\ablation\analysis_$a\robustness
}
```
 
## Full run plan (10 training runs per architecture; results go to D:\paper8\results\journal)
 

```powershell
$J = "D:\paper8\results\journal"
# 1. ResNet: reuse seeds 0-4, train seeds 5-9 (same split, same settings as seeds 0-4)
foreach ($s in 0,1,2,3,4) { New-Item -ItemType Directory -Force "$J\runs_resnet\seed$s" | Out-Null; Copy-Item D:\paper8\results\runs\seed$s\* "$J\runs_resnet\seed$s\" }
foreach ($s in 5,6,7,8,9) { python 05_train.py --patches D:\paper8\results\lidc_patches_v2 --out $J\runs_resnet --seed $s }
# 2. DenseNet: seeds 0-9 at 128 mm with the centre prior
foreach ($s in 0..9) { python 05_train.py --arch densenet --patches D:\paper8\results\lidc_patches_v2 --out $J\runs_densenet --seed $s }
# 3. saliency: IG (3 baselines) + Grad-CAM first, Occlusion in a second pass (512 forward passes per cell and run;
#    it resumes per run and explainer, so it can be stopped and restarted)
foreach ($a in "resnet","densenet") {
  python 06_rider_saliency.py --runs $J\runs_$a --out $J\saliency_$a --explainers ig,ig_blur,ig_mean,gradcam --save-patches
  python 06_rider_saliency.py --runs $J\runs_$a --out $J\saliency_$a --explainers occlusion
}
# 4. analyses per architecture
foreach ($a in "resnet","densenet") {
  python 07_gauge_analysis.py --saliency $J\saliency_$a --out $J\analysis_$a --no-lesion-figure
  python 10_robustness.py    --saliency $J\saliency_$a --out $J\analysis_$a\robustness
  python 14_faithfulness.py  --saliency $J\saliency_$a --runs $J\runs_$a --out $J\analysis_$a\faithfulness
  python 15_scan_registration.py --saliency $J\saliency_$a --out $J\analysis_$a\registration
}
# 5. architecture comparison (same 353 cells, paired by patient)
python 12_compare_ablation.py --arms $J\analysis_resnet $J\analysis_densenet --names resnet densenet --out $J\compare_arch
python 13_ablation_ci.py --a $J\saliency_resnet\measurands.csv --b $J\saliency_densenet\measurands.csv `
    --controls-a $J\analysis_resnet\robustness\controls.csv --controls-b $J\analysis_densenet\robustness\controls.csv `
    --names resnet densenet --out $J\compare_arch\arch_ci.csv
```
 

## Sensitivity analyses: centring, resolution and neutral deletion

About 9 h on the server: lesion-centring sensitivity (06
`--centre-jitter-mm 5/10`, 18 `centring`), layer3 Grad-CAM and resolution-matched comparison (18 `resolution`), and
deletion with a neutral patch-mean replacement, voxel- and cube-level (14 `--fill mean --unit voxel|cube`).

## Additional analyses

- `python 19_size_and_mass.py` (local, CSVs only): LAF vs lesion size, size-adjusted ICC, correct vs missed decisions,
  positive/negative attribution mass -> `results/journal/analysis_*/size_mass/`.
- Server run, ~8 h: 20 (variance-ratio CIs, run offsets,
  1000 refits), 21 (core/rim, shell, negative LAF, ROI radius), 14 deletion controls (`--random matched`,
  `--within lesion`, `--mode insertion`), 09 lesion figures (median and poorest-agreement lesion).

## Figures and validation

- `python 16_journal_figures.py` -> `fig_faithfulness.png` (deletion curves, cascading randomisation).
- `python 08_paper_figures.py --analysis ..\results\journal\analysis_resnet` -> 10-run variance-sources figure.
- `python 17_validate_cmc.py` recomputes the numbers in the manuscript source from `results/journal` and
  `results/ablation` (manuscript folder set with the `PAPER_DIR` environment variable) and writes
  `validation_sheet.md` there; every row must be MATCH.

## Validation sheet

`python 11_validate_paper.py` recomputes every number in `paper/main.tex` from `results/` (values printed only on the
server are in `paper/logged_values.json`) and writes `paper/validation_sheet.md`. Run it after any change to the paper
or the results; every row must be MATCH, CONFIG or LOGGED.

## Paper figures

```bash
python 08_paper_figures.py      # fig_variance_sources.png, fig_icc_forest_seeds.png, table_main.csv (CSVs only)
python 09_lesion_figure.py      # fig_lesion_compact.png: 2 x 4 (IG / Grad-CAM x scan 1-2 at 1.25L, 1.25S);
                                # needs the RIDER DICOM + results/saliency; copy the .png to paper/figs/
```
 
