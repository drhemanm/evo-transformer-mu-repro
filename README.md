# EvoTransformer-MU — Reproducibility (Small-Compute)

Minimal, **single-GPU** PyTorch scripts to reproduce small-compute results for *EvoTransformer-MU* —
an evolving transformer architecture optimized for efficiency and adaptability.

Runs on **Colab T4 16GB** or any single GPU with ~16GB VRAM.

**Preprint DOI:** https://doi.org/10.5281/zenodo.16833210

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Pretrain (MLM) on WikiText-2
python train.py --dataset wikitext --subset wikitext-2-raw-v1 --seq_len 512 --epochs 3 --save ckpt/evo_small.pt --seed 42

python finetune.py --task piqa --seq_len 256 --epochs 3 --ckpt ckpt/evo_small.pt --seed 42

python finetune.py --task hellaswag --seq_len 256 --epochs 3 --ckpt ckpt/evo_small.pt --seed 42

