## Overview

`tiny_decoder.py` is a minimal decoder-only Transformer that implements the same architecture pattern used by GPT, Llama, and OPT — scaled down to a single block with a toy vocabulary. It trains a next-token prediction objective on short sequences, then generates completions autoregressively. The entire pipeline — tokenization, dataset construction, model definition, training loop, and inference — is self-contained in one file with no external dependencies beyond PyTorch.

The architecture: token embeddings + positional embeddings → one Transformer block (causal self-attention → FFN, both with residual connections and layer norm) → linear projection to vocabulary logits. Training builds (prefix, next_token) pairs from each sequence, left-pads prefixes to `max_len`, and optimizes cross-entropy loss.


## Running

source ~/vllm-env/bin/activate

# Default settings
python tiny_decoder.py

# All available options
python tiny_decoder.py \
  --max-len 10 \      # Maximum context length (default: 10)
  --d-model 32 \      # Embedding/hidden dimension (default: 32)
  --d-ff 64 \         # Feed-forward inner dimension (default: 64)
  --n-heads 1 \       # Number of attention heads (default: 1)
  --epochs 600 \      # Training epochs (default: 600)
  --lr 0.03 \         # Learning rate (default: 3e-2)
  --seed 42           # Random seed (default: 42)

## Some useful options

# Longer context, more training
python tiny_decoder.py --max-len 16 --epochs 1000

# Wider model with multi-head attention
python tiny_decoder.py --d-model 64 --d-ff 128 --n-heads 4

# Quick iteration
python tiny_decoder.py --epochs 100

# Reproducibility check — change seed
python tiny_decoder.py --seed 123

Note that --d-model must be divisible by --n-heads, and --max-len must be ≥ 5 (the length of the longest training sequence).

---

## Annotated Function List

### Utilities

**`set_seed(seed)`** — Seeds Python's `random`, PyTorch CPU, and PyTorch CUDA/ROCm RNGs for reproducibility.

**`get_device()`** — Returns `torch.device("cuda")` if a GPU is available (covers both CUDA and ROCm), otherwise CPU.

**`backend_description()`** — Introspects `torch.version.hip` and `torch.version.cuda` to produce a human-readable string like `"ROCm/HIP 7.2.26015 on Radeon RX 7900 XT"`.

### Tokenizer

**`TinyTokenizer.__init__(vocab)`** — Builds bidirectional mappings (`tok2id`, `id2tok`) from a vocabulary list. Validates that `<pad>`, `<bos>`, `<eos>` are present.

**`TinyTokenizer.encode(tokens)`** — Converts a list of string tokens to integer IDs. Raises on unknown tokens (no UNK fallback — strict by design).

**`TinyTokenizer.decode(ids)`** — Converts integer IDs back to a space-joined string, stripping `<pad>` tokens.

### Dataset Construction

**`left_pad(ids, pad_id, max_len)`** — Prepends `<pad>` IDs so the sequence is exactly `max_len` long. This is the opposite of right-padding used in batch training of encoder models — left-padding preserves the causal property that the last position always contains the most recent token.

**`build_dataset(sequences, tokenizer, max_len)`** — Core data preparation. For each training sequence, generates every (prefix → next_token) pair. E.g., `<bos> I like cats <eos>` produces four training examples where progressively longer left-padded prefixes map to their next token. Returns `(x_tensor, y_tensor)` ready for training.

### Model Components

**`CausalSelfAttention.__init__(d_model, n_heads, max_len)`** — Initializes Q/K/V/output projection matrices (all bias-free) and registers a lower-triangular boolean causal mask as a non-persistent buffer.

**`CausalSelfAttention.forward(x)`** — Standard scaled dot-product attention. Projects input to Q/K/V, reshapes to `[batch, heads, seq, head_dim]`, computes attention scores, applies the causal mask (fills upper triangle with `-inf`), softmaxes, multiplies by V, and projects output. This is the manual implementation of what FlashAttention/ROCM_ATTN accelerate at scale.

**`FeedForward.__init__(d_model, d_ff)` / `.forward(x)`** — Two-layer MLP with ReLU activation. `d_model → d_ff → d_model`. In production models this is typically SwiGLU; ReLU here for clarity.

**`TransformerBlock.forward(x)`** — Pre-norm residual block: `x + attn(layernorm(x))` then `x + ffn(layernorm(x))`. This is the pre-LN variant used by GPT-2/Llama (as opposed to post-LN from the original Transformer paper).

**`TinyDecoderTransformer.__init__(vocab_size, cfg)`** — Assembles the full model: token embedding, positional embedding, one TransformerBlock, final layer norm, and a linear head projecting to `vocab_size`. Production models stack N blocks; this uses one.

**`TinyDecoderTransformer.forward(input_ids)`** — Forward pass. Adds token + positional embeddings, passes through the block, layer-norms, and returns logits from the **last position only** (`x[:, -1, :]`). This is the next-token prediction head — it only needs the final position's representation because the causal mask ensures it attends to all prior tokens.

### Training

**`train(model, x, y, device, cfg)`** — Standard PyTorch training loop. Adam optimizer, cross-entropy loss, full-batch gradient descent (no mini-batching — dataset is 8 examples). Logs loss and accuracy at intervals.

### Inference

**`predict_next_token(model, tokenizer, prefix_tokens, device, cfg)`** — Single-step inference. Encodes the prefix, left-pads to `max_len`, forward-passes, and argmaxes the logits to get the predicted next token ID. This is the decode step that production inference engines like vLLM optimize with KV caching.

**`generate(model, tokenizer, prefix_tokens, device, cfg)`** — Autoregressive loop. Repeatedly calls `predict_next_token`, appending each predicted token to the prefix, until `<eos>` or `max_len`. Note: this recomputes the full forward pass on every step — no KV cache. This is exactly the inefficiency that vLLM's paged attention eliminates.

### CLI

**`parse_args()`** — Argparse wrapper exposing `--max-len`, `--d-model`, `--d-ff`, `--n-heads`, `--epochs`, `--lr`, `--seed`.

**`main()`** — Orchestrates everything: parse args → build config → set seed → build tokenizer and dataset → instantiate model → train → generate from four test prompts.

---

## What PyTorch Provides

| PyTorch component | What tiny_decoder uses it for |
|---|---|
| `torch.Tensor` | All data representation — input IDs, weights, logits, loss values |
| `nn.Embedding` | Token and positional embedding lookup tables |
| `nn.Linear` | Q/K/V projections, output projection, FFN layers, vocabulary head |
| `nn.LayerNorm` | Pre-norm in the Transformer block and final norm |
| `nn.Sequential` | Composes the FFN's linear→ReLU→linear stack |
| `F.softmax` | Attention weight normalization after masking |
| `F.cross_entropy` | Training loss — combines log-softmax + NLL in one fused op |
| `torch.tril` | Builds the causal attention mask (lower-triangular boolean matrix) |
| `torch.optim.Adam` | Weight updates during training |
| `tensor.backward()` | Autograd — computes gradients through the entire computation graph |
| `torch.no_grad()` | Disables gradient tracking during inference for memory/speed |
| `register_buffer` | Stores the causal mask on the correct device without treating it as a parameter |
| `torch.cuda.*` | Device detection, GPU placement, RNG seeding (works transparently on ROCm via HIP) |

The key insight: PyTorch provides the autodiff engine (backprop), the GPU kernel dispatch (GEMM, softmax, embedding lookup all run on your 7900 XT via HIP), and the optimizer. Everything else — the architecture, the causal mask logic, the training data construction, the generation loop — is your code built on top of those primitives.

