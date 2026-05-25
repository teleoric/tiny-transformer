## Overview

`tiny_decoder.py` is a minimal decoder-only Transformer that implements the same architecture pattern used by GPT-2, GPT-3/4, Llama, and Mistral — scaled down to a couple of blocks and an eleven-word vocabulary. It trains a next-token prediction objective on a small structured dataset, then demonstrates three distinct generation behaviours (memorisation, composition, distribution). The entire pipeline — tokenisation, dataset construction, model definition, training loop, generation — is self-contained in one file with no external dependencies beyond PyTorch.

The architecture: token embeddings + learned positional embeddings → N pre-norm Transformer blocks (causal multi-head self-attention → GELU FFN, both with residual connections and LayerNorm) → final LayerNorm → tied linear projection to vocabulary logits. Training shifts inputs by one position to produce targets, supervises every position in a single forward pass, and uses `-100` (IGNORE_INDEX) in padded target slots so cross-entropy ignores them with no manual masking.

The dataset encodes a single rule — *the verb determines the object class, independent of the subject* — so the three generation demos at the bottom of `main()` exercise distinct capabilities rather than rote regurgitation.

## Running

```bash
source ~/vllm-env/bin/activate  # or whatever env has PyTorch installed

# Default settings
python tiny_decoder.py

# All available options (defaults shown)
python tiny_decoder.py \
  --max-len 16 \         # Maximum context length
  --d-model 64 \         # Embedding / hidden dimension
  --d-ff 256 \           # Feed-forward inner dimension
  --n-heads 4 \          # Number of attention heads
  --n-layers 2 \         # Number of stacked Transformer blocks
  --dropout 0.1 \        # Dropout applied to embeddings, attention, residuals, FFN
  --epochs 1500 \        # Training epochs (full-batch)
  --lr 3e-3 \            # AdamW learning rate
  --weight-decay 0.0 \   # Applied only to 2D+ weights via parameter-group split
  --grad-clip 1.0 \      # Global grad-norm clip; set to 0 to disable
  --seed 42              # Random seed
  # --no-tie-embeddings  # Untie lm_head.weight from token_emb.weight (default: tied)
```

Constraints: `--d-model` must be divisible by `--n-heads`; `--max-len` must be ≥ the longest training sequence minus 1 (with the default dataset of length-5 sequences, that means ≥ 4).

### Useful variations

```bash
# Deeper, wider model — exercises residual init scaling and multi-head attention
python tiny_decoder.py --n-layers 4 --d-model 128 --d-ff 512 --n-heads 8

# Quick iteration during development
python tiny_decoder.py --epochs 200

# Reproducibility check — change seed
python tiny_decoder.py --seed 123

# Train with weight decay (param-group split: 2D+ weights only)
python tiny_decoder.py --weight-decay 0.01
```

## Tests

```bash
pytest tests/
```

Pins shape, causality (no-future-leak), and dataset shift alignment. Fast — runs in well under a second on CPU.

---

## Annotated Function List

### Utilities

**`set_seed(seed)`** — Seeds Python's `random`, PyTorch CPU, and PyTorch CUDA/ROCm RNGs. Note: full bit-exact determinism additionally requires `torch.use_deterministic_algorithms(True)`; seeding alone is enough for reproducible loss curves.

**`get_device()`** — Returns `torch.device("cuda")` if a GPU is available (covers both CUDA and ROCm), otherwise CPU.

**`backend_description()`** — Introspects `torch.version.hip` and `torch.version.cuda` to produce a human-readable backend string like `"ROCm/HIP 7.2.26015 on Radeon RX 7900 XT"`.

### Configuration

**`Config`** — Frozen dataclass holding model + optimisation + reproducibility settings. `__post_init__` validates at construction time so a bad config fails loudly before training starts. The CLI builds its argparse defaults from `Config()`, keeping a single source of truth.

**`_config_from_args(args)`** — Filters argparse `Namespace` to `Config` fields via `dataclass fields()`, so the CLI can be extended with non-Config flags (`--verbose`, etc.) without breaking Config construction.

### Tokenizer

**`TinyTokenizer.__init__(vocab)`** — Builds bidirectional mappings (`tok2id`, `id2tok`) from a vocabulary list. Validates that `<pad>`, `<bos>`, `<eos>` are present. Stores the input as a fresh list so external mutation can't corrupt internal state.

**`TinyTokenizer.encode(tokens)`** — Converts string tokens to integer IDs. Raises with a clean message on unknown tokens (no UNK fallback — strict by design for a closed vocabulary).

**`TinyTokenizer.decode(ids, skip_special=True)`** — Converts IDs back to a space-joined string, optionally stripping special tokens (`<pad>`, `<bos>`, `<eos>`).

### Dataset Construction

**`build_dataset(sequences, tokenizer, max_len)`** — Core data preparation. Produces the standard GPT shifted-target tensors: `inputs = ids[:-1]` right-padded with `pad_id`, `targets = ids[1:]` right-padded with `IGNORE_INDEX` so cross-entropy skips those slots. One sequence → one row, supervised at every real position in a single forward pass.

### Model Components

**`CausalSelfAttention(d_model, n_heads, dropout)`** — Multi-head scaled dot-product attention with separate Q/K/V/output projections (bias-free, GPT-2/Llama style). The causal mask is *not* owned here — it's passed in from the top-level model, so multi-layer stacks don't duplicate the buffer. Attention and residual dropout applied per nanoGPT.

**`FeedForward(d_model, d_ff, dropout)`** — Two-layer MLP, `d_model → d_ff → d_model` with GELU activation and post-projection dropout. Modern production models often use SwiGLU; GELU is GPT-2's choice and one fewer unfamiliar primitive.

**`TransformerBlock(cfg)`** — Pre-norm decoder block: `x + attn(LN(x))` then `x + ffn(LN(x))`. Pre-norm (LN before each sub-layer) is the GPT-2 / Llama variant and is materially more stable to train at depth than the original post-norm formulation.

**`TinyDecoder(vocab_size, cfg)`** — Full model. Token + learned positional embeddings → embedding dropout → `n_layers` Transformer blocks → final LayerNorm → linear head. Three architectural niceties on top:

- **Weight tying**: `lm_head.weight = token_emb.weight` halves embedding parameters and tends to improve generalisation (toggle with `--no-tie-embeddings`).
- **Shared causal mask**: registered once as a non-persistent buffer on the top-level module, threaded through each block — no per-layer duplication.
- **GPT-2 residual init scaling**: weights of the residual-stream exit projections (`out_proj`, `fc2`) are initialised with std `0.02 / sqrt(2 * n_layers)`, so the residual stream variance stays O(1) as depth grows.

### Training

**`_build_optimizer(model, cfg)`** — AdamW with parameter-group weight decay. 2D+ tensors (Linear/Embedding weights) go into the decay group; biases and LayerNorm scale/shift (all 1D) go into a no-decay group. This is the GPT-2/nanoGPT convention — applying L2 to LN affine params hurts training.

**`train(model, inputs, targets, device, cfg)`** — Full-batch training loop. AdamW, cross-entropy with `ignore_index=IGNORE_INDEX`, optional grad-norm clipping. Logs loss and a mask-aware accuracy every 100 epochs.

### Generation

**`generate(model, tokenizer, prompt, cfg, device, ...)`** — Autoregressive generation. Returns `(prompt_tokens, generated_tokens)` so callers can format the two halves independently. Supports:

- `temperature=0.0` → pure greedy (`argmax`).
- `temperature>0` with optional `top_k` → multinomial sampling.
- `generator=<torch.Generator>` → reproducible sampling decoupled from the global RNG.

Sliding-window truncation past `max_len` is implemented but documented as lossy — with absolute positional embeddings, the same physical token gets a different position embedding once the window slides, and the model wasn't trained on that distribution. RoPE/ALiBi fix this.

**`_run_greedy_demo` / `_run_sampled_demo`** — Render the three prompt categories (memorised, compositional, distributional) to stdout. Results go via `print` rather than the logger because they're user-facing output, not diagnostics.

### CLI

**`parse_args()`** — Argparse wrapper. Defaults come from `Config()`; new fields added to `Config` get picked up automatically.

**`main()`** — Orchestrates everything: parse args → build config → set seed → build tokenizer and dataset → instantiate model → train → run all three generation demos.

---

## Prompt categories

The dataset encodes one rule: **verb determines object class**. `like`/`feed` → pets (`cats`, `dogs`); `see` → wildlife (`birds`, `fish`). Subject is independent.

| Category | What it tests | Example | Expected (greedy) |
|---|---|---|---|
| **Memorised** | Capacity / optimisation: did the model fit the training data? | `<bos> I see` | `birds <eos>` |
| **Compositional** | Generalisation: did the model learn verb → class, or just memorise subject-object correlations? Subject-verb pair never seen together in training. | `<bos> we like` | `cats <eos>` or `dogs <eos>` |
| **Distributional** | Probabilistic next-token modelling: when training has multiple valid completions, does sampling reflect both? | `<bos> I like` × 5 samples | A mix of `cats` and `dogs` |

If memorised prompts fail, the model is underfit (bump `--epochs` or `--n-layers`). If compositional prompts fail but memorised ones succeed, the model overfit subject↔object correlations (try more dropout or a larger dataset). If distributional sampling collapses to one object, temperature is effectively too low or the model has overconfident logits.

---

## What PyTorch Provides

| PyTorch component | What `tiny_decoder` uses it for |
|---|---|
| `torch.Tensor` | All data representation — input IDs, weights, logits, loss values |
| `nn.Embedding` | Token and positional embedding lookup tables |
| `nn.Linear` | Q/K/V/output projections, FFN layers, vocabulary head |
| `nn.LayerNorm` | Pre-norm in each block and final norm before the head |
| `nn.ModuleList` | Container for the stack of Transformer blocks |
| `nn.Dropout` | Embedding, attention, residual, and FFN dropout |
| `F.gelu` | FFN activation |
| `F.softmax` | Attention weight normalisation after masking |
| `F.cross_entropy` | Training loss with `ignore_index` for padded targets |
| `torch.tril` | Lower-triangular boolean causal mask |
| `torch.optim.AdamW` | Decoupled weight-decay optimiser, with parameter-group split |
| `torch.nn.utils.clip_grad_norm_` | Global gradient-norm clipping |
| `torch.Generator` | Reproducible sampling decoupled from global RNG |
| `torch.multinomial` | Categorical sampling for temperature/top-k generation |
| `tensor.backward()` | Autograd through the entire computation graph |
| `torch.no_grad()` | Disables gradient tracking for inference / generation |
| `register_buffer` | Stores the causal mask on the correct device, non-persistent (not in state_dict) |
| `torch.cuda.*` | Device detection, GPU placement, RNG seeding (works on ROCm via HIP) |

The key insight: PyTorch provides the autodiff engine (backprop), the GPU kernel dispatch (GEMM, softmax, embedding lookup all run on your GPU via CUDA or HIP), and the optimiser. Everything else — the architecture, the causal mask logic, the training data construction, the generation loop — is application code built on top of those primitives.
