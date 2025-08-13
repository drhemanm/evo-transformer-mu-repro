import argparse, os, random, csv
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from evo_model import EvoGenome, EvoForMLM

def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def mask_tokens(input_ids, tokenizer, mlm_prob=0.15):
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_prob, device=labels.device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, device=labels.device, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, device=labels.device, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    return input_ids, labels

def collate(batch, tokenizer, seq_len):
    texts = [b["text"] for b in batch]
    toks = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    input_ids = toks["input_ids"]
    attention_mask = toks["attention_mask"]
    input_ids, labels = mask_tokens(input_ids, tokenizer)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="wikitext")
    ap.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default="ckpt/evo_small.pt")
    ap.add_argument("--log", type=str, default="logs/train_mlm.csv")
    args = ap.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    ds = load_dataset(args.dataset, args.subset)
    train_ds = ds["train"].shuffle(seed=args.seed)
    val_ds = ds["validation"]

    g = EvoGenome(d_model=384, n_layers=6, n_heads=6, ffn_mult=4.0,
                  vocab_size=tokenizer.vocab_size, max_len=args.seq_len)
    model = EvoForMLM(g)

    def collate_fn(batch): return collate(batch, tokenizer, args.seq_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    num_update_steps_per_epoch = len(train_dl)
    num_warmup_steps = min(2000, num_update_steps_per_epoch)
    num_train_steps = num_update_steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps, num_train_steps)

    model, opt, train_dl, val_dl, sched = accelerator.prepare(model, opt, train_dl, val_dl, sched)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    def evaluate():
        model.eval()
        total, count = 0.0, 0
        for batch in val_dl:
            with torch.no_grad():
                out = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = out["loss"]
            total += loss.item()
            count += 1
        return total / max(1, count)

    best = float("inf")
    with open(args.log, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch","step","train_loss","val_loss"])
        for epoch in range(1, args.epochs+1):
            model.train()
            for step, batch in enumerate(train_dl, start=1):
                out = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = out["loss"]
                accelerator.backward(loss)
                opt.step(); sched.step(); opt.zero_grad()

                if step % 50 == 0:
                    accelerator.print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
                    writer.writerow([epoch, step, f"{loss.item():.6f}", ""])

            val_loss = evaluate()
            accelerator.print(f"[val] epoch {epoch} loss {val_loss:.4f}")
            writer.writerow([epoch, "end", "", f"{val_loss:.6f}"])

            if accelerator.is_main_process and val_loss < best:
                best = val_loss
                accelerator.print(f"new best {best:.4f}, saving to {args.save}")
                accelerator.save_state(args.save)

if __name__ == "__main__":
    main()
