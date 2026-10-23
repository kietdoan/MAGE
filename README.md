# MAGE: Towards Generalizable Multi-timescale EEG Representations

This repository provides the MAGE model, EEG preprocessing pipelines, pretraining code, downstream training, and multi-scale aggregator analysis. The included downstream launchers provide examples for FACED and Sleep-EDFx using the pretrained checkpoint `ckpts/mage.ckpt`.

## Installation

Run the following commands from this directory (`MAGE/`):

```bash
conda env create -f environment.yml
conda activate mage
```

`environment.yml` specifies Python 3.9, PyTorch 2.1.2, and CUDA 11.8, together with the training and preprocessing dependencies. The GPU examples require a compatible NVIDIA driver. Adjust GPU selection, batch size, and worker counts in the launchers for your machine.

All commands below run from `MAGE/` unless explicitly shown otherwise. Replace `/path/to/...` placeholders with your own paths.

## Directory guide

| Path | Purpose |
| --- | --- |
| `ckpts/mage.ckpt` | Pretrained checkpoint used by the downstream launchers. |
| `environment.yml` | Conda environment specification. |
| `preprocessing/downstream/` | Preparation scripts for FACED, Sleep-EDFx, PhysioNet-MI, SHU-MI, KaggleERN, TUAB, and BCIC2a. |
| `preprocessing/pretrain/` | TUH EEG preparation for LMDB-based pretraining. |
| `pretrain/` | Pretraining model, training entry point, and example launcher. |
| `downstream/` | Dataset loaders, MAGE backbone, task heads, training, and evaluation. |
| `scripts/` | Aggregator evaluation orchestration and CSV/plot utilities. |
| `train*.sh` | FACED and Sleep-EDFx downstream launchers described below. |

Datasets and downstream run results are not included. Training launchers save generated outputs to the sibling directory `../MAGE_private_artifacts/` by default; no existing contents of that directory are required. To choose another training output location, set an absolute path:

```bash
export MAGE_OUTPUT_ROOT=/path/to/experiment_outputs
```

This variable controls training outputs. Aggregator evaluation output paths are configured separately. `.gitignore` excludes common local datasets, logs, and generated checkpoints while retaining `ckpts/mage.ckpt`.

## Prepare the data

Obtain the raw dataset from its provider, then use the corresponding preparation script under `preprocessing/downstream/`. Configure the input and output paths as indicated in that script and run it from the repository root. Use the resulting processed-data directory for downstream training and evaluation.

### Example: FACED

In `preprocessing/downstream/prepare_faced.sh`, set `ROOT_DIR` to the directory containing the subject `.pkl` files and `OUT_DB` to your desired output LMDB directory. Then run:

```bash
bash preprocessing/downstream/prepare_faced.sh
export FACED_PATH=/path/to/processed/FACED
```

Set `FACED_PATH` to the same directory as `OUT_DB` so the FACED training launchers can locate the processed data. Follow the corresponding preparation script's instructions for other datasets.

## Downstream training examples

All root-level training launchers load `ckpts/mage.ckpt`, freeze the encoder, and train for 50 epochs with batch size 64. Multi-scale mode combines predictions from patch sizes **50, 100, and 200** through the aggregator. Single-scale mode selects one scale with `--active_head_index`: **0 → 50**, **1 → 100**, **2 → 200**.

| Launcher | Mode | Folds per invocation | Arguments |
| --- | --- | --- | --- |
| `train_faced_multi_scale.sh` | FACED multi-scale | All 4 sequentially | None |
| `train_multi_scale_single_scale_faced_index0.sh` | FACED scale 0 | All 4 sequentially | None |
| `train_multi_scale_single_scale_faced_index1.sh` | FACED scale 1 | All 4 sequentially | None |
| `train_multi_scale_single_scale_faced_index2.sh` | FACED scale 2 | All 4 sequentially | None |
| `train_sleepedf_multi_scale.sh` | Sleep-EDFx multi-scale | One selected fold, or all 5 with `-1` | `--fold <fold>` |
| `train_sleepedf_single_scale.sh` | Sleep-EDFx single-scale | One selected fold, or all 5 with `-1` | `--scale-index <index> --fold <fold>` |

The launchers provide example configurations that you can adapt to your experiment.

### Run directly on a GPU machine

With `mage` activated and the dataset paths configured:

```bash
# FACED: train all four folds using the multi-scale model.
bash train_faced_multi_scale.sh

# Sleep-EDFx: train only fold 0 using the multi-scale model.
bash train_sleepedf_multi_scale.sh --fold 0

# Sleep-EDFx: train only fold 0 using scale 0.
bash train_sleepedf_single_scale.sh --scale-index 0 --fold 0
```

The `#SBATCH` directives are ignored when a launcher is run with `bash`.

### Run all five Sleep-EDFx folds

To run the five Sleep-EDFx multi-scale folds sequentially:

```bash
for fold in {0..4}; do
  bash train_sleepedf_multi_scale.sh --fold "$fold"
done
```

For the five folds using single scale 0:

```bash
for fold in {0..4}; do
  bash train_sleepedf_single_scale.sh --scale-index 0 --fold "$fold"
done
```

Repeat the second loop with `--scale-index 1` and `--scale-index 2` to evaluate all three single scales: **15 fold runs total**.

Alternatively, the current Python runner accepts fold `-1` to run all five folds sequentially in one invocation:

```bash
bash train_sleepedf_multi_scale.sh --fold -1
# Or all five folds for single scale 0:
bash train_sleepedf_single_scale.sh --scale-index 0 --fold -1
```

Separate fold runs save checkpoints and logs in their own fold directories, but write to the same experiment-level `csv/summary.csv`; that summary is overwritten rather than merged across invocations. Running all five folds together writes a combined summary with a mean row.

## Pretraining from scratch

Pretraining is optional when using the provided downstream checkpoint. Edit `IN_ROOT` and `OUT_DIR` in `preprocessing/pretrain/prepare_pretrain.sh` to select the raw TUH EEG input and processed output locations:

```bash
bash preprocessing/pretrain/prepare_pretrain.sh
```

The script creates `train.lmdb` and `val.lmdb`. Point `DATA_ROOT` to the directory containing both files, then run the pretraining example from `pretrain/`:

```bash
export DATA_ROOT=/path/to/preprocessed/pretrain
(
  cd pretrain
  bash pretrain_example.sh
)
```

## Inspect multi-scale aggregator weights

After downstream training, `scripts/run_mage_aggregator_folders.py` discovers fold checkpoints, runs the test-set exporter, and combines per-sample weights, metrics, plots, and fold/seed statistics. Use multi-scale result folders with this tool.

The orchestrator reads the following processed-data locations:

| Dataset | Environment variable | Default path under the repository |
| --- | --- | --- |
| FACED | `FACED_PATH` | `data/FACED` |
| Sleep-EDFx | `SLEEPEDF_PATH` | `data/SLEEPEDF` |
| PhysioNet-MI | `PHYSIOMI_PATH` | `data/PhysioMI` |
| SHU-MI | `SHUMI_PATH` | `data/SHUMI` |
| KaggleERN | `KAGGLE_ERN_PATH` | `data/KaggleERN` |
| TUAB | `TUAB_PATH` | `data/TUAB` |
| BCIC2a | `BCIC2A_PATH` | `data/BCIC2a` |

For example, to inspect a FACED multi-scale run, set `FACED_PATH` to your processed data directory and run:

```bash
export FACED_PATH=/path/to/processed/FACED
python scripts/run_mage_aggregator_folders.py \
  ../MAGE_private_artifacts/results_seed100/multi_scale_faced \
  --output-dir ../MAGE_private_artifacts/aggregator_eval_multi_seed/faced_mage \
  --device 0
```

Useful options:

- `--dry-run`: list discovered folds and checkpoints without inference.
- `--skip-existing`: reuse existing raw sample CSVs and rerun aggregation.
- `--batch-size` and `--num-workers`: adjust inference resources.
- `--output-dir`: choose the analysis destination; the default is `../MAGE_private_artifacts/aggregator_eval_selected`.

`downstream/test_mage_multi_scale_visualize.py` is the per-checkpoint exporter. `scripts/run_shared_encoder_scale_weights_folders.py` supplies aggregation helpers, and `scripts/aggregate_mage_scale_weights.py` produces dataset-level summaries.
