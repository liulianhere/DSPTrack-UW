# DSPTrack: Dual-Side Purification for Underwater Tracking

Official PyTorch implementation of **DSPTrack**, a single-object underwater tracker that reformulates underwater tracking as a *query–reference co-purification* problem. This codebase is built on top of [OSTrack](https://github.com/botaoye/OSTrack) / [HIPTrack](https://arxiv.org/abs/2311.02072).

## Abstract

Underwater single-object tracking suffers from degraded search regions and noisy initial templates. DSPTrack purifies **both sides** of the relation model before matching:

- **MAFF** (Multi-scale Anisotropic Feature Fusion) rectifies the search-side query and key features into deformation-robust representations.
- **SAAM** (Spatial Adaptive Attention Module, with SCA + DCA) purifies the initial template into a clean matching reference.
- **AGTF** (Attention-Guided Token Filtering) is retained as an auxiliary input-level token gate.
- A **bidirectional cross-verification decoder** suppresses unidirectional false matches.

Trained only on standard tracking data (LaSOT, TrackingNet, GOT-10k), DSPTrack reaches **68.9% AUC on UOT100** and **66.7% AUC on UTB180** with **no underwater-specific training data**.

## Main Results

| Benchmark | Metric | DSPTrack |
|-----------|--------|----------|
| UOT100    | AUC    | 68.9 ± 0.1 |
| UTB180    | AUC    | 66.7 ± 0.3 |
| OTB100    | AUC    | 70.8 ± 0.2 |
| GOT-10k   | AO     | 71.4 ± 0.4 |

Reported as mean ± std over three training seeds. Speed: **42 FPS** on a single NVIDIA RTX 4090.

## Requirements

- Python 3.8
- PyTorch 2.1.2
- CUDA 12.1

```bash
conda create -n dsptrack python=3.8 -y
conda activate dsptrack
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data Preparation

Place the datasets under `./data`:

```
${PROJECT_ROOT}
 -- data
     -- lasot          # training
     -- got10k         # training + evaluation
     -- trackingnet    # training
     -- uot100         # evaluation
     -- utb180         # evaluation
     -- otb100         # evaluation
```

- **Training data**: LaSOT, GOT-10k, TrackingNet
- **Evaluation data**: UOT100, UTB180, OTB100, GOT-10k

> Note: the base OSTrack/HIPTrack code ships dataset loaders for `utb`, `otb`, `got10k_test`, etc. The **UOT100** loader is a custom addition — make sure `lib/test/evaluation/uot100dataset.py` and its entry in `lib/test/evaluation/datasets.py` are present before evaluating UOT100.

## Pretrained Weights

Download the pretrained DSPTrack model and place it under `./output/checkpoints/`:

> TODO: add the download link (GitHub Releases / Zenodo / Google Drive)

## Training

Default training recipe (from the paper):

| Setting | Value |
|---------|-------|
| Template input | 192 × 192 |
| Search input | 384 × 384 |
| Cropping factors | 2 and 5 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Epochs | 100 |
| Samples per epoch | 20,000 |

```bash
# LaSOT + GOT-10k + TrackingNet
python tracking/train.py \
    --script dsptrack \
    --config dsptrack \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node 1

# GOT-10k only
python tracking/train.py \
    --script dsptrack \
    --config dsptrack_got \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node 1
```

> The shipped `experiments/dsptrack/dsptrack.yaml` is the HIPTrack base config (lr 1e-4, batch 32, and an extra `COCO17` training dataset). Update it to the recipe above (lr 3e-4, batch 16, datasets `LASOT` / `GOT10K_vottrain` / `TRACKINGNET` only, `SAMPLE_PER_EPOCH` 20000) to reproduce the reported numbers.

## Evaluation

```bash
# GOT-10k
python tracking/test.py dsptrack dsptrack --dataset_name got10k_test --threads 4

# UTB180
python tracking/test.py dsptrack dsptrack --dataset_name utb --threads 4

# OTB100
python tracking/test.py dsptrack dsptrack --dataset_name otb --threads 4

# UOT100 (requires the custom UOT100 loader)
python tracking/test.py dsptrack dsptrack --dataset_name uot100 --threads 4
```

## Citation

```bibtex
@article{zhu2026dsptrack,
  title   = {DSPTrack: Dual-Side Purification for Underwater Tracking},
  author  = {Zhu, Zhongben and Liu, Lian and Wang, Xiaomin and Zhang, Li and Mu, Xiaokai and Qin, Hongde and Bai, Guiqiang},
  note    = {Under review. Code available at https://github.com/<your-username>/DSPTrack}
}
```

## License

> TODO: add a license (e.g., MIT) if you want to specify one.
