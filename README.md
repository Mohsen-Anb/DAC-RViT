# DAC-RViT: ICE catheter roll-angle estimation from monoplane fluoroscopy

Divide-and-conquer hybrid ResNet-18 / Vision Transformer classifier (Phase 1),
per-bin XGBoost regressors on (cos, sin) targets (Phase 2), and a Confidence
Estimation Pipeline (temperature scaling, meta-classifier, validation-selected
acceptance threshold).

## Layout

```
dac_rvit_core.py        shared library (data, model, metrics, CEP, predictor)
dac_rvit_train.py       training / evaluation pipeline (edit CONFIG at the top)
inference/predict.py    inference with a released model bundle
inference/evaluate_test.py   reproduce the reported test metrics from the bundle
requirements.txt
```

## Data layout

```
<main_folder>/
  <recording_1>/labels.csv, frame_0000.png, frame_0001.png, ...
  <recording_2>/...
```

`labels.csv` columns used: `frame`, `Roll_Angle`, `X_Local_Origin`,
`Y_Local_Origin`, `X_Tip_Displacement`, `Y_Tip_Displacement`
(`Time (ms)` is optional). Frames are the preprocessed 224x224 crops
(inpainted, aligned along the LO-TD axis). No other column is read.

## Running

1. Edit `CONFIG` in `dac_rvit_train.py`: set `main_folder`, `results_root`.
2. Quick end-to-end check: set `smoke_test = True` and run
   `python dac_rvit_train.py`. This uses a small data fraction, two epochs,
   one N value and two seeds, and produces every table and figure.
3. Full run: set `smoke_test = False` (and `data_fraction = 1.0`) and run again.

## Experimental protocol

* Frames are assigned to train/val/test in contiguous blocks of
  `block_len` consecutive frames within each recording (default 15 frames,
  1 s at 15 fps). The fraction of test frames adjacent to a training frame
  is reported in `metrics.csv`.
* All angular errors use the shortest-arc circular distance.
* Number of bins N, the released seed/checkpoint, the temperature, the
  meta-classifier and the acceptance threshold are selected on the
  validation split only. The test split is evaluated once per run.
* Augmentation is photometric only (brightness, contrast, Gaussian noise,
  blur) and is applied to the training split only.
* PNF normalisation statistics are computed on the training split.

## Phase 2 and post-processing options (CONFIG)

| key | effect |
|---|---|
| `neighbor_overlap` | k: regressor i is trained on bins i-k..i+k; 0 = own bin only |
| `soft_top_k` | 1 = hard routing with boundary clipping; >1 = probability-weighted average of the top-k regressor outputs in (cos, sin) space |
| `cep_use_disagreement` | adds the top-k regressor disagreement (circular spread, deg) to the meta-classifier inputs |
| `temporal_window` | causal circular-mean smoothing over the last w frames inside each consecutive test block; also evaluates the hold-last-confident fallback policy (`test_mae_deploy_fallback`) and uncertain-run lengths; 0 = off |
| `label_smoothing`, `augment.erase_prob` | optional regularisation for Phase 1 |

## Outputs

Per run (`<results_root>/<stage>/N<nn>_seed<s>/`):

| file | content |
|---|---|
| `split_manifest.csv` | recording, frame, block id, split for every frame |
| `predictions_{train,val,test}.csv` | per-frame true/pred roll, circular error, bins, P(correct), confident flag |
| `metrics.csv` | all scalar metrics (validation and test) |
| `per_bin_test.csv` | per-bin n, Phase 1 accuracy with 95% CI, MAE |
| `threshold_sweep_val.csv` | validation MAE and coverage per candidate threshold |
| `baselines_and_controls.csv` | shuffled-PNF, global regressors, bin-centre baselines |
| `learning_curves.csv` | per-epoch loss and accuracy |
| `model_bundle/` | everything needed for inference |
| `fig_*.png/.pdf` | learning curves, confidence-vs-error, per-bin accuracy, risk-coverage, reliability, scatter, prediction trace, error histograms, threshold sweep |

Summary (`<results_root>/`):

| file | content |
|---|---|
| `table1_n_sweep.csv`, `fig4_n_sweep` | N sweep, validation-based selection |
| `table_seed_stability.csv`, `table_seed_summary.csv`, `fig_seed_stability` | all seeds at the selected N; released checkpoint marked |
| `table2_ablation.csv`, `fig_ablation` | matched baselines, same splits and metric |
| `table_controls.csv` | image-only, PNF-only, shuffled-PNF |
| `all_runs.csv` | one row per training run |
| `released_model/` | the released checkpoint bundle with its manifest, predictions and metrics |

## Inference

```
python inference/predict.py --bundle <results_root>/released_model \
    --image frame_0012.png --lo 60.40 46.65 --td 70.74 50.06

python inference/evaluate_test.py --bundle <results_root>/released_model \
    --data <main_folder>
```

`evaluate_test.py` re-runs the released model on the exact test frames listed
in the bundled manifest and recomputes every metric; the numbers must match
`released_model/metrics.csv`.
