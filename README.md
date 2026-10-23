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

This variable controls training outputs. Slurm log paths and aggregator evaluation output paths are configured separately. `.gitignore` excludes common local datasets, logs, and generated checkpoints while retaining `ckpts/mage.ckpt`.

## Prepare the data

Obtain the raw datasets from their providers. For launchers under `preprocessing/downstream/`, edit the input/output placeholders before running them. Use the processed output directory as the corresponding training or evaluation data path.

### FACED

In `preprocessing/downstream/prepare_faced.sh`, set `ROOT_DIR` to the directory of subject `.pkl` files and `OUT_DB` to the output LMDB directory. Then run:

```bash
bash preprocessing/downstream/prepare_faced.sh
export FACED_PATH=/path/to/processed/FACED
```

### Sleep-EDFx

The Sleep-EDFx preparation script uses Braindecode's `SleepPhysionet` loader and writes preprocessed `.pt` windows into class subdirectories. Set the destination before running it; the same variable is used by the training launchers:

```bash
export SLEEPEDF_PATH=/path/to/processed/SLEEPEDF
bash preprocessing/downstream/prepare_sleepedf.sh
```

### Other datasets

Preparation launchers are also provided as `prepare_physiomi.sh`, `prepare_shumi.sh`, `prepare_kaggle_ern.sh`, `prepare_tuab.sh`, and `prepare_bcic2a.sh` under `preprocessing/downstream/`. Edit their paths and invoke them from the repository root.

The unified downstream entry point supports these datasets, but dedicated training shell examples are currently provided only for FACED and Sleep-EDFx. Inspect its options with:

```bash
python downstream/run_downstream_full.py --help
```

Pass dataset paths explicitly when invoking this Python entry point, for example `--physiomi_path /path/to/processed/PhysioMI`. It does not read the dataset environment variables used by the shell launchers and aggregator orchestrator.

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

### Run on a Slurm cluster

Activate `mage` before submission. Create the log directory before calling `sbatch`, because Slurm opens log files before executing the script:

```bash
mkdir -p ../MAGE_private_artifacts/logs
sbatch --partition=YOUR_PARTITION train_faced_multi_scale.sh
```

Replace `YOUR_PARTITION` with your cluster's GPU partition, or omit the option if a suitable default is configured. The launchers request one GPU, 10 CPUs, and 24 hours; adjust these directives for your cluster. Override `--output` and `--error` when submitting if you want a different Slurm log location.

For five separate Sleep-EDFx multi-scale jobs:

```bash
for fold in {0..4}; do
  sbatch --partition=YOUR_PARTITION train_sleepedf_multi_scale.sh --fold "$fold"
done
```

For five separate jobs using single scale 0:

```bash
for fold in {0..4}; do
  sbatch --partition=YOUR_PARTITION train_sleepedf_single_scale.sh --scale-index 0 --fold "$fold"
done
```

Repeat the second loop with `--scale-index 1` and `--scale-index 2` to evaluate all three single scales: **15 jobs total**.

Alternatively, the current Python runner accepts fold `-1` to run all five folds sequentially in one invocation:

```bash
bash train_sleepedf_multi_scale.sh --fold -1
# Or all five folds for single scale 0:
bash train_sleepedf_single_scale.sh --scale-index 0 --fold -1
```

Allow enough runtime if submitting an all-fold invocation to Slurm. Separate fold runs save checkpoints and logs in their own fold directories, but write to the same experiment-level `csv/summary.csv`; that summary is overwritten rather than merged across invocations. Running all five folds together writes a combined summary with a mean row.

### Training outputs

Under `MAGE_OUTPUT_ROOT` (default `../MAGE_private_artifacts`), the launchers use:

```text
results_seed100/
├── multi_scale_faced/
├── single_scale_faced/
├── multi_scale_sleepedf/
└── single_scale_sleepedf/
```

Additional subdirectories encode the dataset, model settings, split, and selected head. Within each fold, `ckpts/` stores checkpoints, `tb/` stores TensorBoard logs, and `csv/` stores logged metrics. Single-scale experiments include a `Head0`, `Head1`, or `Head2` directory.

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

The pretraining example uses 50 epochs, 19 channels, and patch sizes 50, 100, and 200. Its outputs go to the `pretrain/` subdirectory of `MAGE_OUTPUT_ROOT`, or `../MAGE_private_artifacts/pretrain/` relative to the repository root by default. Pretraining and downstream launchers have independent settings.

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

The FACED and Sleep-EDFx training shell launchers use the same respective variables and defaults. Use absolute paths for overrides.

For a FACED run saved in the default output location:

```bash
python scripts/run_mage_aggregator_folders.py \
  ../MAGE_private_artifacts/results_seed100/multi_scale_faced \
  --output-dir ../MAGE_private_artifacts/aggregator_eval_multi_seed/faced_mage \
  --device 0
```

For Sleep-EDFx, replace the input directory with `../MAGE_private_artifacts/results_seed100/multi_scale_sleepedf` and select a different output directory. If you changed `MAGE_OUTPUT_ROOT`, pass the corresponding input path explicitly.

The expected checkpoint layout is:

```text
<results_dir>/.../<DATASET>/.../<split_dir>/foldX/ckpts/best-*.ckpt
```

The tool prefers one `best-*.ckpt` per fold and falls back to `last.ckpt`; multiple best checkpoints in one fold cause an error. It recognizes split directories `4fold` for FACED/TUAB, `5fold` for SLEEPEDF/SHUMI/KaggleERN, `K5fold_SUBJ_fixT4` for PhysioMI, and `LOSO` for BCIC2a. Fold names `valFoldX` and `sX_*` are also recognized.

Pass multiple result directories to pool experiments. Seeds are inferred from a `results_seed<N>` ancestor, defaulting to 100. The orchestrator passes that inferred value as both `--seed` and `--val_seed` to the exporter; check this against your training configuration when matching validation splits.

Useful options:

- `--dry-run`: list discovered folds and checkpoints without inference.
- `--skip-existing`: reuse existing raw sample CSVs and rerun aggregation.
- `--batch-size` and `--num-workers`: adjust inference resources.
- `--output-dir`: choose the analysis destination; the default is `../MAGE_private_artifacts/aggregator_eval_selected`.

`downstream/test_mage_multi_scale_visualize.py` is the per-checkpoint exporter. `scripts/run_shared_encoder_scale_weights_folders.py` supplies aggregation helpers, and `scripts/aggregate_mage_scale_weights.py` produces dataset-level summaries.
