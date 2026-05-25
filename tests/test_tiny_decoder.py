"""
Smoke tests for tiny_decoder.

Run with:
    pytest tests/

These tests are deliberately tiny - they pin down properties that any
refactor of the core architecture is likely to break (shape, causality,
shift alignment). They are not a substitute for end-to-end training
verification, which `python tiny_decoder.py` provides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make the repo root importable without installing the package, so
# `pytest tests/` works from a fresh clone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiny_decoder import (  # noqa: E402
    BOS_TOKEN,
    EOS_TOKEN,
    IGNORE_INDEX,
    SPECIAL_TOKENS,
    Config,
    TinyDecoder,
    TinyTokenizer,
    build_dataset,
)


def _toy_setup() -> tuple[Config, TinyTokenizer]:
    """Small, fast, dropout-free config for deterministic testing."""
    cfg = Config(
        max_len=8,
        d_model=16,
        d_ff=32,
        n_heads=2,
        n_layers=2,
        dropout=0.0,
        epochs=1,
    )
    vocab = list(SPECIAL_TOKENS) + ["a", "b", "c"]
    tok = TinyTokenizer(vocab)
    return cfg, tok


# -----------------------------------------------------------------------------
# Forward-pass shape
# -----------------------------------------------------------------------------
def test_forward_shape() -> None:
    cfg, tok = _toy_setup()
    model = TinyDecoder(tok.vocab_size, cfg)
    x = torch.zeros(2, cfg.max_len, dtype=torch.long)
    assert model(x).shape == (2, cfg.max_len, tok.vocab_size)


def test_forward_shorter_than_max_len() -> None:
    """Inputs shorter than max_len should work (variable T at inference)."""
    cfg, tok = _toy_setup()
    model = TinyDecoder(tok.vocab_size, cfg)
    short_T = cfg.max_len // 2
    x = torch.zeros(1, short_T, dtype=torch.long)
    assert model(x).shape == (1, short_T, tok.vocab_size)


def test_forward_rejects_too_long() -> None:
    cfg, tok = _toy_setup()
    model = TinyDecoder(tok.vocab_size, cfg)
    x = torch.zeros(1, cfg.max_len + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds max_len"):
        model(x)


# -----------------------------------------------------------------------------
# Init invariants
# -----------------------------------------------------------------------------
def test_tied_embeddings_stay_tied() -> None:
    """
    After __init__ runs, lm_head.weight must share storage with
    token_emb.weight. Two things could plausibly break this:

      * `self.apply(_init_weights)` reassigning weights in a way that
        breaks identity (it doesn't — `nn.init.normal_` is in-place).
      * The residual-scaling loop matching `lm_head.weight` (it doesn't —
        the name suffix filter only catches `out_proj.weight` and
        `fc2.weight`).

    `data_ptr()` rather than `is` because PyTorch can wrap the same
    storage in different `Parameter` objects in edge cases; storage
    identity is what actually matters for tying.
    """
    cfg, tok = _toy_setup()
    model = TinyDecoder(tok.vocab_size, cfg)
    assert model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()


# -----------------------------------------------------------------------------
# Causality
# -----------------------------------------------------------------------------
def test_no_future_leak() -> None:
    """
    Logits at position t must not depend on inputs at positions > t.

    This is the property the causal mask exists to enforce. If the mask
    were dropped or reversed, this test fails - which is exactly when
    you want to know.
    """
    cfg, tok = _toy_setup()
    model = TinyDecoder(tok.vocab_size, cfg).eval()

    base = torch.tensor([[tok.bos_id, 3, 4, 5, 0, 0, 0, 0]])
    perturbed = base.clone()
    # Change only positions 5..7 (the "future" relative to position 4).
    perturbed[0, 5:] = torch.tensor([6, 7, 6])

    with torch.no_grad():
        l_base = model(base)[0, :5]
        l_pert = model(perturbed)[0, :5]

    # Positions 0..4 must be bit-identical because nothing they can attend to changed.
    assert torch.allclose(l_base, l_pert, atol=1e-6)


# -----------------------------------------------------------------------------
# Dataset shift alignment
# -----------------------------------------------------------------------------
def test_dataset_shift_alignment() -> None:
    """
    For a sequence of length L, build_dataset produces:
        x = ids[:-1]            then right-padded with pad_id
        y = ids[1:]             then right-padded with IGNORE_INDEX
    so for positions 0..L-3 we should have y[t] == x[t+1] (the shift
    invariant). The final supervised position L-2 has y == EOS, which
    is NOT what x[L-1] holds (that's the first pad), so the invariant
    does not extend to it.
    """
    cfg, tok = _toy_setup()
    seqs = [[BOS_TOKEN, "a", "b", "c", EOS_TOKEN]]  # length 5
    x, y = build_dataset(seqs, tok, cfg.max_len)

    # Shift invariant for the interior of the real sequence (positions 0..2).
    assert torch.equal(y[0, :3], x[0, 1:4])

    # Final supervised position is the EOS target.
    assert y[0, 3].item() == tok.eos_id

    # Everything past the real sequence is IGNORE_INDEX in targets and pad in inputs.
    assert (y[0, 4:] == IGNORE_INDEX).all()
    assert (x[0, 4:] == tok.pad_id).all()


def test_dataset_rejects_too_long() -> None:
    """Sequences longer than max_len + 1 cannot fit after shift-by-one."""
    cfg, tok = _toy_setup()
    too_long = [BOS_TOKEN] + ["a"] * cfg.max_len + [EOS_TOKEN]  # length max_len + 2
    with pytest.raises(ValueError, match="exceeds max_len"):
        build_dataset([too_long], tok, cfg.max_len)


# -----------------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------------
def test_config_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        Config(d_model=17, n_heads=4)


def test_config_rejects_negative_weight_decay() -> None:
    with pytest.raises(ValueError, match="weight_decay"):
        Config(weight_decay=-0.1)
