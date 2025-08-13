import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class EvoGenome:
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_mult: float = 4.0
    dropout: float = 0.1
    vocab_size: int = 30522
    max_len: int = 512

class EvoEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(ffn_mult * d_model)),
            nn.GELU(),
            nn.Linear(int(ffn_mult * d_model), d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(a)
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x

class EvoEncoder(nn.Module):
    def __init__(self, g: EvoGenome):
        super().__init__()
        self.g = g
        self.token_emb = nn.Embedding(g.vocab_size, g.d_model)
        self.pos_emb = nn.Embedding(g.max_len, g.d_model)
        self.blocks = nn.ModuleList([
            EvoEncoderBlock(g.d_model, g.n_heads, g.ffn_mult, g.dropout)
            for _ in range(g.n_layers)
        ])
        self.norm = nn.LayerNorm(g.d_model)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x

class EvoForMLM(nn.Module):
    def __init__(self, g: EvoGenome):
        super().__init__()
        self.enc = EvoEncoder(g)
        self.lm_head = nn.Linear(g.d_model, g.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.enc(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        return {"loss": loss, "logits": logits}

class EvoForMC(nn.Module):
    """Multiple-choice head: mean-pool then classify among choices."""
    def __init__(self, g: EvoGenome):
        super().__init__()
        self.enc = EvoEncoder(g)
        self.classifier = nn.Linear(g.d_model, 1)

    def forward(self, input_ids, attention_mask):
        B, C, T = input_ids.shape
        flat_ids = input_ids.view(B*C, T)
        flat_mask = attention_mask.view(B*C, T)
        h = self.enc(flat_ids, attention_mask=flat_mask)
        mask = flat_mask.unsqueeze(-1).float()
        h = (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        logits = self.classifier(h).view(B, C)
        return logits
