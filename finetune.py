import argparse, os, random, csv
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from evo_model import EvoGenome, EvoForMC
from sklearn.metrics import accuracy_score

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def collate_piqa(batch, tokenizer, seq_len):
    # PIQA fields: goal, sol1, sol2, label
    contexts = [b["goal"] for b in batch]
    choices = [[b["sol1"], b["sol2"]] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    flat_first = [c for pair in [[ctx, ctx] for ctx in contexts] for c in pair]
    flat_second = [opt for pair in choices for opt in pair]
    toks = tokenizer(flat_first, flat_second, padding="max_length", truncation=True,
                     max_length=seq_len, return_tensors="pt")
    C = 2
    input_ids = toks["input_ids"].view(len(labels), C, -1)
    attention_mask = toks["attention_mask"].view(len(labels), C, -1)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def collate_hellaswag(batch, tokenizer, seq_len):
    # HellaSwag fields: ctx, endings (list[str] of length 4), label
    contexts = [b["ctx"] for b in batch]
    endings = [b["endings"] for b in batch]  # list of 4 strings
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    flat_first = [ctx for ctx in contexts for _ in range(len(endings[0]))]
    flat_second = [e for ends in endings for e in ends]
    toks = tokenizer(flat_first, flat_second, padding="max_length", truncation=True,
                     max_length=seq_len, return_tensors="pt")
    C = len(endings[0])
    input_ids = toks["input_ids"].view(len(labels), C, -1)
    attention_mask = toks["attention_mask"].view(len(labels), C, -1)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, choices=["piqa","hellaswag"], required=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", type=str, required=True, help="path to weights .pt saved by train.py")
    ap.add_argument("--log", type=str, default="logs/finetune_mc.csv")
    args = ap.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if args.task == "piqa":
        ds = load_dataset("piqa")
        train_ds, val_ds = ds["train"], ds["validation"]
        collate_fn = lambda b: collate_piqa(b, tokenizer, args.seq_len)
    else:
        ds = load_dataset("hellaswag")
        train_ds, val_ds = ds["train"], ds["validation"]
        collate_fn = lambda b: collate_hellaswag(b, tokenizer, args.seq_len)

    g = EvoGenome(d_model=384, n_layers=6, n_heads=6, ffn_mult=4.0,
                  vocab_size=tokenizer.vocab_size, max_len=args.seq_len)
    model = EvoForMC(g)

    # ✅ Load model weights saved by train.py BEFORE accelerator.prepare
    accelerator.print(f"Loading weights from {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(f"Loaded with missing={missing}, unexpected={unexpected}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    num_update_steps_per_epoch = len(train_dl)
    num_warmup = min(1000, num_update_steps_per_epoch)
    num_train_steps = num_update_steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, num_warmup, num_train_steps)

    model, opt, train_dl, val_dl, sched = accelerator.prepare(model, opt, train_dl, val_dl, sched)

    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    def evaluate():
        model.eval()
        correct = 0; total = 0
        for batch in val_dl:
            with torch.no_grad():
                logits = model(batch["input_ids"], batch["attention_mask"])
                preds = logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += preds.numel()
        return correct / max(1, total)

    with open(args.log, "w", newline="") as fcsv:
        writer = csv.writer(fcsv); writer.writerow(["epoch","step","train_loss","val_acc"])
        for epoch in range(1, args.epochs+1):
            model.train()
            for step, batch in enumerate(train_dl, start=1):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
                accelerator.backward(loss)
                opt.step(); sched.step(); opt.zero_grad()
                if step % 50 == 0:
                    accelerator.print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
                    writer.writerow([epoch, step, f"{loss.item():.6f}", ""])
            acc = evaluate()
            accelerator.print(f"[val] epoch {epoch} acc {acc:.4f}")
            writer.writerow([epoch, "end", "", f"{acc:.6f}"])

if __name__ == "__main__":
    main()
