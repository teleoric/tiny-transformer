#!/usr/bin/env python3
"""
tiny_decoder_transformer.py

A tiny, trainable, decoder-only Transformer with configurable max_len.

Features:
- arbitrary max_len via Config / CLI
- single Transformer block
- causal self-attention
- next-token training objective built from left-padded prefixes
- works on CPU / CUDA / ROCm without code changes

Default training data:
    <bos> I like cats <eos>
    <bos> you like dogs <eos>

Example usage:
    python tiny_decoder_transformer.py
    python tiny_decoder_transformer.py --max-len 10
    python tiny_decoder_transformer.py --max-len 16 --epochs 1000
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Configuration
# ----------------------------

@dataclass(frozen=True)
class Config:
    max_len: int = 10
    d_model: int = 32
    d_ff: int = 64
    n_heads: int = 1
    epochs: int = 600
    lr: float = 3e-2
    seed: int = 42


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def backend_description() -> str:
    if not torch.cuda.is_available():
        return "CPU"

    name = torch.cuda.get_device_name(0)
    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)

    if hip_version:
        return f"ROCm/HIP {hip_version} on {name}"
    if cuda_version:
        return f"CUDA {cuda_version} on {name}"
    return f"GPU on {name}"


# ----------------------------
# Tokenizer
# ----------------------------

class TinyTokenizer:
    def __init__(self, vocab: Sequence[str]) -> None:
        if not vocab:
            raise ValueError("Vocabulary must not be empty.")

        self.vocab: List[str] = list(vocab)
        self.tok2id: Dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2tok: Dict[int, str] = {i: tok for i, tok in enumerate(self.vocab)}

        required = ["<pad>", "<bos>", "<eos>"]
        missing = [tok for tok in required if tok not in self.tok2id]
        if missing:
            raise ValueError(f"Vocabulary is missing required tokens: {missing}")

        self.pad = self.tok2id["<pad>"]
        self.bos = self.tok2id["<bos>"]
        self.eos = self.tok2id["<eos>"]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for tok in tokens:
            if tok not in self.tok2id:
                raise KeyError(f"Unknown token: {tok}")
            ids.append(self.tok2id[tok])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        toks = [self.id2tok[int(i)] for i in ids if int(i) != self.pad]
        return " ".join(toks)


# ----------------------------
# Dataset construction
# ----------------------------

def left_pad(ids: Sequence[int], pad_id: int, max_len: int) -> List[int]:
    if len(ids) > max_len:
        raise ValueError(f"Sequence too long: {len(ids)} > max_len={max_len}")
    return [pad_id] * (max_len - len(ids)) + list(ids)


def build_dataset(
    sequences: Sequence[Sequence[str]],
    tokenizer: TinyTokenizer,
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (prefix -> next token) training pairs.

    Example:
        <bos> I like cats <eos>

    Produces:
        [pad pad pad pad <bos>]      -> I
        [pad pad pad <bos> I]        -> like
        [pad pad <bos> I like]       -> cats
        [pad <bos> I like cats]      -> <eos>
    """
    xs: List[List[int]] = []
    ys: List[int] = []

    for seq in sequences:
        ids = tokenizer.encode(seq)

        if len(ids) > max_len:
            raise ValueError(
                f"Training sequence length {len(ids)} exceeds max_len={max_len}: {seq}"
            )

        for i in range(1, len(ids)):
            prefix = ids[:i]
            target = ids[i]
            xs.append(left_pad(prefix, tokenizer.pad, max_len))
            ys.append(target)

    if not xs:
        raise ValueError("No training examples were produced.")

    x_tensor = torch.tensor(xs, dtype=torch.long)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return x_tensor, y_tensor


# ----------------------------
# Model
# ----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seq_len, :seq_len]
        att = att.masked_fill(~causal, torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_len: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_len=max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyDecoderTransformer(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        self.block = TransformerBlock(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            max_len=cfg.max_len,
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        returns logits for the final position: [B, vocab_size]
        """
        if input_ids.ndim != 2:
            raise ValueError(f"Expected rank-2 input_ids, got shape {tuple(input_ids.shape)}")

        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_len:
            raise ValueError(
                f"Input length {seq_len} exceeds max_len={self.cfg.max_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.block(x)
        x = self.ln_f(x)

        last = x[:, -1, :]
        logits = self.head(last)
        return logits


# ----------------------------
# Training / inference
# ----------------------------

def train(
    model: TinyDecoderTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    cfg: Config,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()

    x = x.to(device)
    y = y.to(device)

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0 or epoch == cfg.epochs:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean().item()
            print(f"epoch={epoch:4d}  loss={loss.item():.4f}  acc={acc:.3f}")


@torch.no_grad()
def predict_next_token(
    model: TinyDecoderTransformer,
    tokenizer: TinyTokenizer,
    prefix_tokens: Sequence[str],
    device: torch.device,
    cfg: Config,
) -> str:
    model.eval()

    prefix_ids = tokenizer.encode(prefix_tokens)
    if len(prefix_ids) > cfg.max_len:
        raise ValueError(
            f"Prefix length {len(prefix_ids)} exceeds max_len={cfg.max_len}"
        )

    padded = left_pad(prefix_ids, tokenizer.pad, cfg.max_len)
    x = torch.tensor([padded], dtype=torch.long, device=device)
    logits = model(x)
    next_id = int(logits.argmax(dim=-1).item())
    return tokenizer.id2tok[next_id]


@torch.no_grad()
def generate(
    model: TinyDecoderTransformer,
    tokenizer: TinyTokenizer,
    prefix_tokens: Sequence[str],
    device: torch.device,
    cfg: Config,
) -> str:
    model.eval()
    out = list(prefix_tokens)

    while len(out) < cfg.max_len:
        next_tok = predict_next_token(
            model=model,
            tokenizer=tokenizer,
            prefix_tokens=out,
            device=device,
            cfg=cfg,
        )
        out.append(next_tok)
        if next_tok == "<eos>":
            break

    return " ".join(out)


# ----------------------------
# CLI / main
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny decoder-only Transformer with configurable max_len."
    )
    parser.add_argument("--max-len", type=int, default=10, help="Maximum context length.")
    parser.add_argument("--d-model", type=int, default=D_MODEL_DEFAULT(), help="Model width.")
    parser.add_argument("--d-ff", type=int, default=D_FF_DEFAULT(), help="Feed-forward width.")
    parser.add_argument("--n-heads", type=int, default=N_HEADS_DEFAULT(), help="Number of attention heads.")
    parser.add_argument("--epochs", type=int, default=600, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=3e-2, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def D_MODEL_DEFAULT() -> int:
    return 32


def D_FF_DEFAULT() -> int:
    return 64


def N_HEADS_DEFAULT() -> int:
    return 1


def main() -> None:
    args = parse_args()

    cfg = Config(
        max_len=args.max_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    device = get_device()

    print(f"backend: {backend_description()}")
    print(f"device:  {device}")
    print(f"config:  {cfg}")

    vocab = [
        "<pad>", "<bos>", "<eos>",
        "I", "you", "like", "cats", "dogs",
    ]
    tokenizer = TinyTokenizer(vocab)

    # These are still short toy sequences. If cfg.max_len > 5, the model can handle
    # the longer context structurally, but the training data remains short unless you
    # add longer examples here.
    sequences = [
        ["<bos>", "I", "like", "cats", "<eos>"],
        ["<bos>", "you", "like", "dogs", "<eos>"],
    ]

    x, y = build_dataset(sequences, tokenizer, cfg.max_len)

    print("training examples:")
    for x_row, y_val in zip(x.tolist(), y.tolist()):
        print(f"  x={tokenizer.decode(x_row)}  ->  y={tokenizer.id2tok[int(y_val)]}")

    model = TinyDecoderTransformer(len(vocab), cfg).to(device)
    train(model, x, y, device, cfg)

    print("\nGeneration:")
    prompts = [
        ["<bos>", "I"],
        ["<bos>", "you"],
        ["<bos>", "I", "like"],
        ["<bos>", "you", "like"],
    ]
    for prompt in prompts:
        generated = generate(model, tokenizer, prompt, device, cfg)
        print(f"  prompt={' '.join(prompt):<18} -> {generated}")


if __name__ == "__main__":
    main()


