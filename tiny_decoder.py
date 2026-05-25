#!/usr/bin/env python3
"""
tiny_decoder.py
===============

A compact decoder-only Transformer small enough to read end-to-end but
architected the way a real GPT-style model is. The goal is pedagogy
without misleading shortcuts.

Design decisions (and why):

  * Single forward pass per sequence with parallel next-token loss.
    Real GPT training computes logits at every position in one pass and
    supervises each with the next token (shifted-by-one targets).

  * Right-padding with -100 ignore index in targets. Left-padding plus
    absolute positional embeddings is a subtle bug: position 0 is
    sometimes <pad>, sometimes <bos>, and the model has to learn to
    untangle that. Right-padding keeps position semantics stable.

  * Pre-norm Transformer block, GELU, tied embeddings, dropout,
    parameter-group weight decay, GPT-2 residual init scaling - the
    defaults you'd find in a small GPT-2 / nanoGPT-style implementation.

  * Causal mask owned by the top-level model and threaded through the
    blocks. Avoids duplicating an [max_len, max_len] buffer per layer.

Things deliberately kept simple for readability:

  * No KV cache during generation. Production decoders cache K/V across
    autoregressive steps for O(T) generation; this code re-encodes
    each step for O(T^2). Comments mark where the cache would hook in.

  * LayerNorm, not RMSNorm. Either is fine; LayerNorm is GPT-2's choice
    and is one less unfamiliar primitive.

  * Manual scaled dot-product attention rather than
    `F.scaled_dot_product_attention`. The unfused version is what you
    want to read while learning; the fused kernel is what you want in
    production.

  * Learned absolute positional embeddings, not RoPE/ALiBi. These are
    the simplest scheme but the reason sliding-window generation
    degrades past max_len.

  * Full-batch training (no DataLoader, no batching). Fine for a dozen
    sequences; obviously not how you would train at scale.

Run:
    python tiny_decoder.py
    python tiny_decoder.py --max-len 16 --epochs 2000 --n-heads 4 --n-layers 2

Works on CPU / CUDA / ROCm without modification.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass, fields
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Module-level constants
# =============================================================================
# Module-level constants are the single source of truth. `Config` defaults
# pick them up; the CLI takes its defaults from `Config()`; no parallel
# definitions to keep in sync.

DEFAULT_MAX_LEN: int = 16
DEFAULT_D_MODEL: int = 64
DEFAULT_D_FF: int = 256
DEFAULT_N_HEADS: int = 4
DEFAULT_N_LAYERS: int = 2
DEFAULT_DROPOUT: float = 0.1
DEFAULT_EPOCHS: int = 1500
DEFAULT_LR: float = 3e-3
DEFAULT_WEIGHT_DECAY: float = 0.0
DEFAULT_GRAD_CLIP: float = 1.0
DEFAULT_SEED: int = 42

# Sentinel value PyTorch's cross_entropy treats as "no loss here". Anything
# strictly < 0 works; -100 is the documented default and the convention
# used throughout the ecosystem (HuggingFace, torchtune, etc).
IGNORE_INDEX: int = -100

# Special tokens. Centralised so swapping vocabularies stays consistent.
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS: Tuple[str, ...] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)

logger = logging.getLogger("tiny_decoder")


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class Config:
    """
    Immutable training/model configuration.

    Frozen so it is hashable, safe to share, and obviously not mutated
    mid-run. The CLI constructs a Config by overriding individual
    fields; the dataclass defaults remain authoritative.
    """

    # Model architecture
    max_len: int = DEFAULT_MAX_LEN
    d_model: int = DEFAULT_D_MODEL
    d_ff: int = DEFAULT_D_FF
    n_heads: int = DEFAULT_N_HEADS
    n_layers: int = DEFAULT_N_LAYERS
    dropout: float = DEFAULT_DROPOUT
    tie_embeddings: bool = True

    # Optimisation
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    grad_clip: float = DEFAULT_GRAD_CLIP

    # Reproducibility
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        # Validate at construction so a bad config fails loudly here rather
        # than five layers deep in a forward pass.
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})"
            )
        if self.max_len < 2:
            raise ValueError(f"max_len must be >= 2 (got {self.max_len})")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1) (got {self.dropout})")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive (got {self.lr})")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1 (got {self.n_layers})")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1 (got {self.epochs})")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0 (got {self.weight_decay})")
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip must be >= 0 (got {self.grad_clip})")


# =============================================================================
# Environment helpers
# =============================================================================
def set_seed(seed: int) -> None:
    """
    Seed all RNGs we touch.

    Note: full bit-exact determinism additionally requires
    `torch.use_deterministic_algorithms(True)` and disabling cuDNN
    benchmarking, which costs performance. Seeding alone is sufficient
    for run-to-run reproducibility of the metrics we log. Sampled
    generation is reproducible via a dedicated `torch.Generator` (see
    `generate`), independently of the global RNG.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick the best available device. ROCm presents itself as 'cuda' in PyTorch."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def backend_description() -> str:
    """Human-readable string describing the active compute backend."""
    if not torch.cuda.is_available():
        return "CPU"

    name = torch.cuda.get_device_name(0)
    # ROCm builds of PyTorch set `torch.version.hip`; mainline CUDA builds set
    # `torch.version.cuda`. The same `torch.cuda.*` API works for both.
    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)
    if hip_version:
        return f"ROCm/HIP {hip_version} on {name}"
    if cuda_version:
        return f"CUDA {cuda_version} on {name}"
    return f"GPU on {name}"


# =============================================================================
# Tokenizer
# =============================================================================
class TinyTokenizer:
    """
    Closed-vocabulary word-level tokenizer.

    For a real model you'd use BPE / SentencePiece / tiktoken. For a
    learning toy, an explicit list of tokens keeps every step
    inspectable.
    """

    def __init__(self, vocab: Sequence[str]) -> None:
        if not vocab:
            raise ValueError("Vocabulary must not be empty.")

        # list() to take ownership: callers should not be able to mutate our
        # internal state by mutating their input afterwards.
        self.vocab: List[str] = list(vocab)
        self.tok2id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2tok = {i: tok for tok, i in self.tok2id.items()}

        missing = [t for t in SPECIAL_TOKENS if t not in self.tok2id]
        if missing:
            raise ValueError(f"Vocabulary missing required special tokens: {missing}")

        self.pad_id = self.tok2id[PAD_TOKEN]
        self.bos_id = self.tok2id[BOS_TOKEN]
        self.eos_id = self.tok2id[EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        # `from None` suppresses the inner KeyError traceback - the rewritten
        # message is more useful and the original adds no information.
        try:
            return [self.tok2id[t] for t in tokens]
        except KeyError as e:
            raise KeyError(f"Unknown token: {e.args[0]!r}") from None

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        out: List[str] = []
        for i in ids:
            tok = self.id2tok[int(i)]
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return " ".join(out)


# =============================================================================
# Dataset construction
# =============================================================================
def build_dataset(
    sequences: Sequence[Sequence[str]],
    tokenizer: TinyTokenizer,
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the standard GPT next-token training tensors.

    Given a sequence  [<bos>, I, like, cats, <eos>]  (ids [1, 3, 5, 6, 2]),
    we emit a single training pair:

        inputs  = [<bos>, I,    like, cats,  <pad>, <pad>, ...]   (length max_len)
        targets = [I,     like, cats, <eos>, -100,  -100,  ...]   (length max_len)

    Notes:

      * Targets are inputs shifted left by one position. Combined with a
        causal mask in the model, this means every position is supervised
        in one forward pass - the correct GPT training objective.

      * Right padding (not left padding) keeps positional embeddings
        consistent: position 0 is always the start of a real sequence.

      * IGNORE_INDEX in the pad target slots makes `cross_entropy` skip
        them with no manual masking.

      * One sequence -> one row. The old code emitted (len-1) rows per
        sequence and did (len-1) forwards for the same supervision signal.

    Returns:
        inputs:  LongTensor  [N, max_len]
        targets: LongTensor  [N, max_len]  (IGNORE_INDEX in pad slots)
    """
    inputs: List[List[int]] = []
    targets: List[List[int]] = []

    for seq in sequences:
        ids = tokenizer.encode(seq)
        if len(ids) < 2:
            raise ValueError(f"Sequence too short to form a target: {seq}")
        if len(ids) > max_len + 1:
            # We slice to ids[:-1] (length L-1), which must fit in max_len.
            # So the longest sequence we can accept is max_len + 1.
            raise ValueError(
                f"Sequence length {len(ids)} exceeds max_len+1={max_len + 1}: {seq}"
            )

        # Shift-by-one: input is everything but last, target is everything but first.
        x_ids = ids[:-1]
        y_ids = ids[1:]

        pad_amount = max_len - len(x_ids)
        x_ids = x_ids + [tokenizer.pad_id] * pad_amount
        y_ids = y_ids + [IGNORE_INDEX]     * pad_amount

        inputs.append(x_ids)
        targets.append(y_ids)

    if not inputs:
        raise ValueError("No training examples were produced.")

    return (
        torch.tensor(inputs,  dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


# =============================================================================
# Model: multi-head causal self-attention
# =============================================================================
class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal mask.

    "Causal" means position t may attend only to positions 0..t. That's
    the only structural difference from a BERT-style encoder block.

    Implementation notes:

      * Q, K, V projections kept as three separate Linears for clarity.
        GPT-2 fuses them into one Linear of size 3*d_model and splits
        afterwards (one matmul instead of three). Functionally identical.

      * The causal mask is owned by the top-level model and passed in,
        so we don't allocate one buffer per block. Slight coupling cost
        in the signature; large win in memory at scale.

      * For inference, K and V at past positions are constant across
        autoregressive steps and should be cached. We do not cache here
        (see module docstring). The hook would be a `past_kv` arg to
        forward and a `(out, new_kv)` return.

      * In production prefer `F.scaled_dot_product_attention` which
        dispatches to Flash Attention on supported hardware. Here we
        keep the unfused form so the math is visible.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # `bias=False` matches GPT-2 / LLaMA. Biases on these projections
        # add parameters with no measurable benefit on top of LayerNorm.
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]; causal_mask: [max_len, max_len] bool
        # returns: [B, T, D]
        B, T, D = x.shape

        # Project to Q, K, V then reshape to per-head: [B, H, T, head_dim].
        # The transpose places the head axis before T so subsequent matmuls
        # broadcast naturally over heads.
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: [B, H, T, T].
        # Scale by 1/sqrt(head_dim) so the variance of the dot products
        # stays O(1) as head_dim grows (otherwise softmax saturates).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask: disallowed positions -> -inf, so softmax gives 0.
        # Slice the (max_len, max_len) mask down to the actual T so the same
        # buffer serves variable-length inference.
        mask = causal_mask[:T, :T]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention weights, then collapse heads back into D.
        out = attn @ v                                  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous()          # [B, T, H, head_dim]
        out = out.view(B, T, D)                         # [B, T, D]

        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """
    Position-wise feed-forward MLP: D -> d_ff -> D.

    The second sub-layer of a Transformer block. The MLP is applied
    identically and independently at each position - all cross-position
    interaction happens inside attention.

    GELU is the modern choice (GPT-2 onward). ReLU works but tends to
    leave more dead units in small models.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Standard pre-norm decoder block.

        x -> LN -> Attention -> + residual ->
          -> LN -> FeedForward -> + residual

    Pre-norm (LN before each sub-layer) is materially easier to train at
    depth than the original post-norm formulation from "Attention Is
    All You Need", and is what every modern GPT-style model uses.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff  = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal_mask)
        x = x + self.ff  (self.ln2(x))
        return x


class TinyDecoder(nn.Module):
    """
    A small GPT-style decoder-only Transformer.

        input_ids -> token_emb + pos_emb -> dropout
                  -> N x TransformerBlock (sharing one causal mask)
                  -> final LayerNorm
                  -> lm_head -> logits over vocab

    The model returns logits at every position, so a single forward pass
    is enough to compute next-token loss for an entire sequence.
    """

    def __init__(self, vocab_size: int, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb   = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb     = nn.Embedding(cfg.max_len, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying: input embedding and output projection encode the same
        # vocabulary, so sharing them halves embedding parameters and tends
        # to improve generalisation. Standard in GPT-2/3.
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        # Single causal mask shared across all blocks. Registered as a buffer
        # so it moves with `.to(device)` but is not a learned parameter, and
        # persistent=False keeps it out of state_dict (re-derivable from cfg).
        mask = torch.tril(torch.ones(cfg.max_len, cfg.max_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

        # GPT-2 init: small normal for Linear/Embedding, identity-ish for LN.
        self.apply(self._init_weights)

        # GPT-2 residual scaling. Each block sums two sub-layer outputs into
        # the residual stream, so after N blocks the stream is the sum of
        # 2*N contributions. Without rescaling the exit projections
        # (out_proj, fc2), the residual variance grows ~sqrt(2*N) with
        # depth and training becomes unstable. Reference: GPT-2 paper §2.3.
        residual_std = 0.02 * (2 * cfg.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]   ->   logits: [B, T, V]
        if input_ids.ndim != 2:
            raise ValueError(
                f"Expected rank-2 input_ids, got shape {tuple(input_ids.shape)}"
            )
        B, T = input_ids.shape
        if T > self.cfg.max_len:
            raise ValueError(
                f"Input length {T} exceeds max_len={self.cfg.max_len}"
            )

        # Absolute positions [0, 1, ..., T-1], broadcast over the batch.
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x, self.causal_mask)
        x = self.ln_f(x)
        return self.lm_head(x)


# =============================================================================
# Training
# =============================================================================
def _build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """
    AdamW with parameter-group weight decay.

    Decay applies to 2D+ weight tensors (Linear and Embedding weights).
    Excluded: biases, LayerNorm scale/shift, and any other 1D parameter.
    Applying L2 to LN affine params or biases hurts training: it pulls
    the LN scale away from its useful range and adds noise to biases
    that the network compensates for elsewhere. This split is the
    GPT-2 / nanoGPT convention.

    Note on tied embeddings: when lm_head.weight is tied to
    token_emb.weight, `named_parameters()` returns the shared tensor
    once (PyTorch deduplicates by identity), so it lands in the decay
    group exactly once.
    """
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # ndim < 2 catches biases and LayerNorm.{weight,bias}.
        # Embedding.weight is 2D and falls into the decay group.
        if p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )


def train(
    model: TinyDecoder,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    cfg: Config,
) -> None:
    """
    Full-batch training loop.

    Real training would use a DataLoader, mini-batching, an LR schedule,
    mixed precision, gradient accumulation, and something like wandb /
    tensorboard for logging. None of that adds insight on a dozen
    sequences, so we keep the loop minimal.
    """
    optimizer = _build_optimizer(model, cfg)

    inputs  = inputs.to(device)
    targets = targets.to(device)
    model.train()

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)                       # [B, T, V]

        # Flatten (batch, time) for cross_entropy. The IGNORE_INDEX we
        # wrote into targets makes the loss skip pad positions, so this
        # is mathematically identical to a hand-written masked loss.
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            targets.view(B * T),
            ignore_index=IGNORE_INDEX,
        )

        loss.backward()

        # Grad clipping is cheap insurance against the occasional spike.
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        if epoch == 1 or epoch % 100 == 0 or epoch == cfg.epochs:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                supervised = targets != IGNORE_INDEX
                acc = (preds[supervised] == targets[supervised]).float().mean().item()
            logger.info(
                "epoch=%4d  loss=%.4f  acc=%.3f", epoch, loss.item(), acc
            )


# =============================================================================
# Generation
# =============================================================================
@torch.no_grad()
def generate(
    model: TinyDecoder,
    tokenizer: TinyTokenizer,
    prompt: Sequence[str],
    cfg: Config,
    device: torch.device,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[str], List[str]]:
    """
    Autoregressive generation with optional temperature / top-k sampling.

    Returns (prompt_tokens, generated_tokens) so callers can format the
    two halves independently without re-encoding the prompt.

    Loop:
        1. Truncate the running sequence to the last max_len tokens.
        2. Forward pass; take logits at the final position.
        3. Apply temperature; optionally restrict to top-k.
        4. argmax (temperature=0) or sample via multinomial.
        5. Append; stop on <eos> or token budget.

    Sliding-window note: when the running sequence exceeds max_len we
    truncate to the most recent max_len tokens. With absolute positional
    embeddings this has a subtle cost - the same physical token receives
    a different position embedding on the next step, and the model was
    trained on token-position pairs that this distribution shift
    violates. Quality therefore degrades immediately past max_len. This
    is one of the main motivations for RoPE and ALiBi, which are
    position-encoding schemes invariant (or near-invariant) to absolute
    offset.

    KV-cache note: steps 1-2 recompute K, V for every past position on
    every step. Caching them across steps drops generation cost from
    O(T^2) to O(T). The hook point is CausalSelfAttention.forward.

    Reproducibility: pass a `torch.Generator` (on the same device as
    the model) to make sampled generation deterministic regardless of
    global RNG state. Greedy generation (temperature=0) is already
    deterministic.
    """
    model.eval()

    if max_new_tokens is None:
        max_new_tokens = cfg.max_len - len(prompt)

    prompt_ids = tokenizer.encode(prompt)
    out_ids: List[int] = list(prompt_ids)

    for _ in range(max_new_tokens):
        context = out_ids[-cfg.max_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)

        logits = model(x)                       # [1, T, V]
        last_logits = logits[0, -1]             # [V]

        if temperature == 0.0:
            # Pure greedy. Skip temperature/top-k entirely for clarity;
            # `generator` is ignored on this path.
            next_id = int(last_logits.argmax().item())
        else:
            scaled = last_logits / temperature

            if top_k is not None and top_k > 0:
                # Keep only the top-k logits; set the rest to -inf so softmax
                # gives them zero probability mass.
                top_values, _ = torch.topk(scaled, k=min(top_k, scaled.size(-1)))
                cutoff = top_values[-1]
                scaled = torch.where(
                    scaled < cutoff,
                    torch.full_like(scaled, float("-inf")),
                    scaled,
                )

            probs = F.softmax(scaled, dim=-1)
            next_id = int(
                torch.multinomial(probs, num_samples=1, generator=generator).item()
            )

        out_ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break

    return (
        [tokenizer.id2tok[i] for i in prompt_ids],
        [tokenizer.id2tok[i] for i in out_ids[len(prompt_ids):]],
    )


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    """
    The CLI takes its defaults from `Config()` so the dataclass remains
    the single source of truth.
    """
    defaults = Config()
    p = argparse.ArgumentParser(
        description="Train a tiny decoder-only Transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-len",       type=int,   default=defaults.max_len)
    p.add_argument("--d-model",       type=int,   default=defaults.d_model)
    p.add_argument("--d-ff",          type=int,   default=defaults.d_ff)
    p.add_argument("--n-heads",       type=int,   default=defaults.n_heads)
    p.add_argument("--n-layers",      type=int,   default=defaults.n_layers)
    p.add_argument("--dropout",       type=float, default=defaults.dropout)
    p.add_argument("--epochs",        type=int,   default=defaults.epochs)
    p.add_argument("--lr",            type=float, default=defaults.lr)
    p.add_argument("--weight-decay",  type=float, default=defaults.weight_decay)
    p.add_argument("--grad-clip",     type=float, default=defaults.grad_clip)
    p.add_argument("--seed",          type=int,   default=defaults.seed)
    p.add_argument(
        "--no-tie-embeddings",
        dest="tie_embeddings",
        action="store_false",
        default=defaults.tie_embeddings,
        help="Disable tying lm_head.weight to token_emb.weight.",
    )
    return p.parse_args()


def _config_from_args(args: argparse.Namespace) -> Config:
    """
    Build a Config from argparse Namespace, ignoring any extra fields.

    Filtering through `dataclass fields()` keeps the CLI safe to extend
    with non-Config flags (--verbose, --save-path, etc) without breaking
    Config construction.
    """
    cfg_kwargs = {
        f.name: getattr(args, f.name)
        for f in fields(Config)
        if hasattr(args, f.name)
    }
    return Config(**cfg_kwargs)


# =============================================================================
# Prompt categories (pedagogical)
# =============================================================================
# The dataset encodes a single rule: the *verb* determines the object class,
# independent of the subject.
#     like, feed -> {cats, dogs}    (pets)
#     see        -> {birds, fish}   (wildlife)
# The three prompt categories below each test a different capability.

# Memorized: prompts that appear verbatim in training with a unique
# completion. Greedy decoding should recover the training row. This is
# a capacity / optimization sanity check - if these fail, the model
# didn't fit.
MEMORIZED_PROMPTS: Tuple[Tuple[str, ...], ...] = (
    (BOS_TOKEN, "I",    "see"),
    (BOS_TOKEN, "you",  "see"),
    (BOS_TOKEN, "we",   "see"),
    (BOS_TOKEN, "they", "see"),
)

# Compositional: subject-verb pairs that never co-occur in training. The
# model must combine "verb -> object class" learned from other rows with
# the novel subject. Success here means the model abstracted the verb
# rule rather than memorising subject-object correlations.
COMPOSITIONAL_PROMPTS: Tuple[Tuple[str, ...], ...] = (
    (BOS_TOKEN, "we",   "like"),
    (BOS_TOKEN, "they", "like"),
    (BOS_TOKEN, "I",    "feed"),
    (BOS_TOKEN, "you",  "feed"),
)

# Distributional: prompts where the training set contains multiple valid
# completions. With temperature>0 sampling, repeated draws should
# produce a mix - demonstrating that the model learned a *distribution*
# over next tokens, not a deterministic mapping.
DISTRIBUTIONAL_PROMPTS: Tuple[Tuple[str, ...], ...] = (
    (BOS_TOKEN, "I",    "like"),
    (BOS_TOKEN, "you",  "like"),
    (BOS_TOKEN, "we",   "feed"),
    (BOS_TOKEN, "they", "feed"),
)


def _run_greedy_demo(
    title: str,
    description: str,
    prompts: Sequence[Sequence[str]],
    model: TinyDecoder,
    tokenizer: TinyTokenizer,
    cfg: Config,
    device: torch.device,
) -> None:
    print(f"\n{title}")
    print(f"  {description}")
    for prompt in prompts:
        p_toks, g_toks = generate(
            model, tokenizer, prompt, cfg, device, temperature=0.0
        )
        print(f"    {' '.join(p_toks):<22} -> {' '.join(g_toks)}")


def _run_sampled_demo(
    title: str,
    description: str,
    prompts: Sequence[Sequence[str]],
    model: TinyDecoder,
    tokenizer: TinyTokenizer,
    cfg: Config,
    device: torch.device,
    n_samples: int = 5,
    temperature: float = 1.0,
) -> None:
    # Dedicated generator: reproducible across runs at fixed seed, and
    # decoupled from the global RNG so it doesn't perturb anything else.
    sampler = torch.Generator(device=device).manual_seed(cfg.seed)

    print(f"\n{title}")
    print(f"  {description}")
    for prompt in prompts:
        # Sample only the immediate next token (the verb's object) for
        # each draw - that's the position where the distribution lives.
        samples: List[str] = []
        for _ in range(n_samples):
            _, g_toks = generate(
                model, tokenizer, prompt, cfg, device,
                temperature=temperature,
                generator=sampler,
                max_new_tokens=1,
            )
            samples.append(g_toks[0] if g_toks else "<empty>")
        print(f"    {' '.join(prompt):<22} -> {samples}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    args = parse_args()
    cfg = _config_from_args(args)

    set_seed(cfg.seed)
    device = get_device()

    logger.info("backend: %s", backend_description())
    logger.info("device:  %s", device)
    logger.info("config:  %s", cfg)

    # Toy vocabulary. Real models use BPE/SentencePiece over tens of thousands
    # of tokens; here, eleven words is enough to demonstrate the rule below.
    vocab = list(SPECIAL_TOKENS) + [
        "I", "you", "we", "they",
        "like", "feed", "see",
        "cats", "dogs", "birds", "fish",
    ]
    tokenizer = TinyTokenizer(vocab)

    # Pattern: verb determines object class.
    #   like/feed -> {cats, dogs}     (pets)
    #   see       -> {birds, fish}    (wildlife)
    # Subject is independent of object class; the model must learn that
    # the verb, not the subject, governs the object distribution.
    sequences = [
        [BOS_TOKEN, "I",    "like", "cats",  EOS_TOKEN],
        [BOS_TOKEN, "I",    "like", "dogs",  EOS_TOKEN],
        [BOS_TOKEN, "you",  "like", "cats",  EOS_TOKEN],
        [BOS_TOKEN, "you",  "like", "dogs",  EOS_TOKEN],
        [BOS_TOKEN, "we",   "feed", "cats",  EOS_TOKEN],
        [BOS_TOKEN, "we",   "feed", "dogs",  EOS_TOKEN],
        [BOS_TOKEN, "they", "feed", "cats",  EOS_TOKEN],
        [BOS_TOKEN, "they", "feed", "dogs",  EOS_TOKEN],
        [BOS_TOKEN, "I",    "see",  "birds", EOS_TOKEN],
        [BOS_TOKEN, "you",  "see",  "fish",  EOS_TOKEN],
        [BOS_TOKEN, "we",   "see",  "birds", EOS_TOKEN],
        [BOS_TOKEN, "they", "see",  "fish",  EOS_TOKEN],
    ]
    inputs, targets = build_dataset(sequences, tokenizer, cfg.max_len)

    logger.info("training examples:")
    for x_row, y_row in zip(inputs.tolist(), targets.tolist()):
        # Render IGNORE_INDEX slots as '_' so the shifted-target structure is visible.
        y_str = " ".join(
            tokenizer.id2tok[i] if i != IGNORE_INDEX else "_" for i in y_row
        )
        logger.info("  x=%s  y=%s", tokenizer.decode(x_row, skip_special=False), y_str)

    model = TinyDecoder(tokenizer.vocab_size, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model parameters: %d", n_params)

    train(model, inputs, targets, device, cfg)

    # Results go to stdout (print) rather than the logger - they're the
    # program's user-facing output, not diagnostic logging.
    _run_greedy_demo(
        title="Memorized prompts (greedy)",
        description=(
            "Each prompt has a unique training completion. Greedy "
            "decoding should reproduce it."
        ),
        prompts=MEMORIZED_PROMPTS,
        model=model, tokenizer=tokenizer, cfg=cfg, device=device,
    )

    _run_greedy_demo(
        title="Compositional prompts (greedy)",
        description=(
            "Subject-verb pair never seen in training. The model must "
            "apply verb -> object-class learned from other subjects."
        ),
        prompts=COMPOSITIONAL_PROMPTS,
        model=model, tokenizer=tokenizer, cfg=cfg, device=device,
    )

    _run_sampled_demo(
        title="Distributional prompts (sampled, temperature=1.0)",
        description=(
            "Training has multiple valid completions. 5 samples should "
            "reflect both objects in the verb's class."
        ),
        prompts=DISTRIBUTIONAL_PROMPTS,
        model=model, tokenizer=tokenizer, cfg=cfg, device=device,
    )


if __name__ == "__main__":
    main()
