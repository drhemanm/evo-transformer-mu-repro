# Model Card — EvoTransformer-MU (Small-Compute Reproducibility Pack)

## Model Details

**Model Name:** EvoTransformer-MU (Small)  
**Version:** v0.1.0  
**Author:** Dr. Heman Mohabeer  
**License:** MIT  
**DOI:** [10.5281/zenodo.16833210](https://doi.org/10.5281/zenodo.16833210)  

EvoTransformer-MU is a **live-evolving Transformer architecture** that incorporates evolutionary principles
— mutation, selection, and shape-safe weight inheritance — directly into the model’s training and adaptation
process. This version is designed for **small-compute reproducibility** on single GPUs (e.g., NVIDIA T4 16GB),
matching the experiments reported in the whitepaper.

---

## Intended Uses & Limitations

**Intended Uses:**
- Research into neuroevolution and adaptive transformer architectures.
- Reproducibility validation for the EvoTransformer whitepaper.
- Educational demonstrations of evolutionary model design.

**Limitations:**
- This small model is **not** a production-ready LLM.
- Trained only on small datasets (WikiText-2, PIQA, HellaSwag) for demonstration purposes.
- Does not include large-scale pretraining or task specialization.

---

## Training Data

- **WikiText-2** (language modeling)
- **PIQA** (physical commonsense reasoning)
- **HellaSwag** (commonsense inference)

All datasets are public and loaded from [HuggingFace Datasets](https://huggingface.co/datasets).

---

## Training Procedure

**Pretraining:**  
- MLM objective on WikiText-2.
- AdamW optimizer, cosine schedule with warmup.
- Sequence length: 512.

**Fine-tuning:**  
- Multiple-choice head for PIQA and HellaSwag.
- Sequence length: 256.
- Accuracy evaluated on validation sets.

---

## Evaluation Results (Small-Compute)

| Task       | Metric    | Result |
|------------|-----------|--------|
| PIQA       | Accuracy  | ~0.68  |
| HellaSwag  | Accuracy  | ~0.54  |
| WikiText-2 | Val Loss  | ~2.3   |

*(Results will vary slightly depending on seed and hardware.)*

---

## Reproducibility

**Seeds:**
 13
 21
 42

---


##Citation
Mohabeer, H. (2025). EvoTransformer: A Whitepaper on Evolving Neural Architectures for Open-Ended AI (Version V1). Zenodo. https://doi.org/10.5281/zenodo.16833210


### Run Commands
```bash
pip install -r requirements.txt

# Pretrain
python train.py --dataset wikitext --subset wikitext-2-raw-v1 --seq_len 512 --epochs 3 --save ckpt/evo_small.pt --seed 42

# Fine-tune (PIQA)
python finetune.py --task piqa --seq_len 256 --epochs 3 --ckpt ckpt/evo_small.pt --seed 42

# Fine-tune (HellaSwag)
python finetune.py --task hellaswag --seq_len 256 --epochs 3 --ckpt ckpt/evo_small.pt --seed 42



