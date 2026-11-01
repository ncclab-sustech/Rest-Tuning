<h1 align="center">Rest-Tuning</h1>

<p align="center">
  <b>Data-Efficient Adaptation of EEG Foundation Models to Individuals<br>via Resting-State Signals</b>
</p>

<p align="center">
  <a href="NeurIPS2026_Rest_Tuning.pdf">Paper</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#data-preparation">Data</a> ·
  <a href="#experiments">Experiments</a>
</p>

Rest-Tuning personalizes an EEG foundation model using a subject's resting-state recordings, then adapts the personalized backbone to downstream tasks with limited labeled data. This repository provides preprocessing, Rest calibration, downstream adaptation, and task-only baselines for **LaBraM, BIOT, CBraMod, and EEGPT** on **M3CV** and **CogBCI**.

<p align="center">
  <img src="figures/Rest-Tuning.jpg" alt="Rest-Tuning: resting-state calibration followed by subject-specific downstream adaptation" width="100%">
</p>

## Installation

The experiments use the **EEGPT** Conda environment: Python **3.10.19**, PyTorch **2.5.1**, and CUDA **12.1**. To create a matching environment:

```bash
conda create -n EEGPT python=3.10.19
conda activate EEGPT
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Run all commands below from the repository root.

## Pretrained checkpoints

Download the checkpoints into `../checkpoints/`, alongside the repository.

| Backbone | Filename | Official download |
| --- | --- | --- |
| LaBraM | `labram-base.pth` | [Download](https://raw.githubusercontent.com/935963004/LaBraM/main/checkpoints/labram-base.pth) |
| BIOT | `EEG-six-datasets-18-channels.ckpt` | [Download](https://raw.githubusercontent.com/ycq091044/BIOT/main/pretrained-models/EEG-six-datasets-18-channels.ckpt) |
| CBraMod | `pretrained_weights.pth` | [Download](https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth) |
| EEGPT | `eegpt_mcae_58chs_4s_large4E.ckpt` | [Author release](https://figshare.com/s/e37df4f8a907a866df4b) |

For EEGPT, extract the file at `Files/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt` in the release.

## Data preparation

Download [M3CV](https://www.kaggle.com/competitions/eeg-biometric-competition/data) and [CogBCI](https://zenodo.org/records/6874129). M3CV expects the original MAT directories with `Enrollment_Info.csv` and `Calibration_Info.csv`; CogBCI expects the EEGLAB recordings under `Original_Data`.

```bash
# M3CV: resting state, four individual tasks, and multitask data.
for task in rest motor p300 ts sss multitask; do
  python "preprocessing/m3cv/${task}.py" \
    --dataset-root /path/to/M3CV --output-dir ../data/m3cv
done

# CogBCI: session 1, single-task and multitask data.
for task in single_task multitask; do
  python "preprocessing/cogbci/${task}.py" \
    --raw-dir /path/to/CogBCI/Original_Data \
    --output-dir ../data/cogbci --session 1 --n-jobs 4
done
```

The standard processed files are:

```text
../data/
├── m3cv/     # rest.h5, motor.h5, p300.h5, ts.h5, sss.h5, multitask.h5
└── cogbci/   # rest.h5, nback.h5, matb.h5, multitask.h5
```

The main protocol uses 1-second epochs at 200 Hz: 64 channels for M3CV and 62 for CogBCI. HDF5 files contain EEG, labels, subject IDs, and task IDs for multitask experiments.

<details>
<summary>Processing variants</summary>

CogBCI `single_task.py` produces `rest.h5` with two rest labels; `multitask.py` also produces `rest_conditions.h5` with four rest conditions. These files use different filtering and label definitions. For sessions 2 and 3, multitask outputs have `_s2` and `_s3` suffixes; use separate output directories for single-task sessions.

The original LaBraM `m3cv_full_0.3.sh` uses two historical variants, retained under concise names:

| Original data file | New relative path |
| --- | --- |
| `processed_M3CV_session1_unified/m3cv_P300_Session1_temporal.h5` | `../data/m3cv/p300_unified.h5` |
| `m3cv_Transient_Sensory.h5` | `../data/m3cv/ts_sensory.h5` |

These variants must be supplied separately. The current processors write `p300.h5` and `ts.h5`; use them only when that processing protocol is intended.

</details>

## Experiments

Scripts are grouped by backbone in `run_sh/`. Each directory provides `COG_full_0.3.sh`, `COG_lora_0.7.sh`, `m3cv_full_0.3.sh`, and `m3cv_lora_0.7.sh`; `0.3` and `0.7` denote the Rest masking ratio. Downstream adaptation supports full fine-tuning, LoRA, and linear probing.

Set `cuda_device` and `recon_path` in the selected script, then launch it from the repository root. For example:

```bash
bash run_sh/CBraMod/COG_full_0.3.sh
bash run_sh/BIOT/COG_lora_0.7.sh
```

Scripts retain the original active experiment blocks and parameters. Some run only downstream adaptation and require an existing Rest checkpoint. Set `--recon_results_dir` to the Rest stage's `--output_root`, including any subdirectory such as `/test`.

The original LaBraM baseline scripts are also included. `COG_baseline.sh` requires an experiment-specific checkpoint at `../checkpoints/labram_base_resume_${subject_id}/checkpoint.pth`. Python entry points in `train/` expose all options through `--help`, including `--subjects` for selecting individuals. Results are written under `result/`.

## Repository structure

```text
Rest-Tuning/
├── README.md
├── NeurIPS2026_Rest_Tuning.pdf
├── figures/Rest-Tuning.jpg
├── requirements.txt
├── src/                     # Models, losses, data loading, and training
├── train/                   # Rest, downstream, and baseline entry points
├── preprocessing/
│   ├── m3cv/
│   └── cogbci/
└── run_sh/                  # LaBraM, BIOT, CBraMod, and EEGPT scripts
```

## Acknowledgments

This work builds on [LaBraM](https://github.com/935963004/LaBraM), [BIOT](https://github.com/ycq091044/BIOT), [CBraMod](https://github.com/wjq-learning/CBraMod), and [EEGPT](https://github.com/BINE022/EEGPT). We thank the authors for making their code and pretrained models available.
