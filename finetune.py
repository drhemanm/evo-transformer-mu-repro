import argparse, os, random, csv
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from evo_model import EvoGenome, EvoForMC
from sklearn.metrics import accuracy_score

def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def prepare_mc_batch(batch, tokenizer, seq_len, task):
    if task == "piqa":
        first_sentences = [[q] * 2 for q in batch["goal"]]
        second_sentences = [[batch["sol1"][i], batch["sol2"][i]] for i in range(len(batch["goal"]))]
    elif task == "hellaswag":
        first_sentences = [[ctx] * len(batch["endings"][i]) for i, ctx in enumerate(batch["ctx_a"])]
        second_sentences = [batch["endings"][i] for i in range(len(batch["ctx_a"]))]
    else:
        raise ValueError("Unsupported task")

    flat_first = sum(first_sentences, [])
    flat_second = sum(second_sentences, [])

    toks = tokenizer(flat_first, flat_second, truncation=True, padding="max_length",
                     max_length=seq_len, return_tensors="pt")
    input_ids = toks["input_ids"].view(len(batch["label"]), -1, seq_len)
    attention_mask = toks["attention_mask"].view(len(batch["label"]), -1, seq_len)
    labels = torch.tensor(batch["label"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="piqa", choices=["piqa", "hellaswag"])
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--log", type=str, default="logs/finetune_mc.csv")
    args = ap.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    ds = load_dataset(args.task)
    train_ds = ds["train"].shuffle(seed=args.seed)
    val_ds = ds["validation"]

    g = EvoGenome(d_model=384, n_layers=6, n_heads=6, ffn_mult=4.0,
                  vocab_size=tokenizer.vocab_size, max_len=args.seq_len)
    model = EvoForMC(g)

    accelerator.print(f"Loading weights from {args.ckpt}")
    accelerator.load_state(args.ckpt)

    def collate_fn(batch): return prepare_mc_batch(batch, tokenizer, args.seq_len, args.task)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    num_update_steps_per_epoch = len(train_dl)
    num_warmup_steps = min(500, num_update_steps_per_epoch)
    num_train_steps = num_update_steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps, num_train_steps)

    model, opt, train_dl, val_dl, sched = accelerator.prepare(model, opt, train_dl, val_dl, sched)

    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    def evaluate():
        model.eval()
        all_preds, all_labels = [], []
        for batch in val_dl:
            with torch.no_grad():
                logits = model(batch["input_ids"], batch["attention_mask"])
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
        acc = accuracy_score(all_labels, all_preds)
        return acc

    with open(args.log, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch","step","train_loss","val_acc"])
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

            val_acc = evaluate()
            accelerator.print(f"[val] epoch {epoch} acc {val_acc:.4f}")
            writer.writerow([epoch, "end", "", f"{val_acc:.6f}"])

if __name__ == "__main__":
    main()
