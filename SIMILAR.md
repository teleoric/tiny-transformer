# When to stop using tiny_decoder

`tiny_decoder.py` is an architecture explainer. It exists so that an engineer can read every line of a GPT-style decoder in one sitting and understand what each piece does. The training loop, dataset, and CLI on top of it are scaffolding — they exist to make the architecture come alive for a toy vocabulary, not to be a general-purpose training framework.

The natural next question is *can I train it on real data?* The honest answer: **the architecture can; the file can't.** And the work required to close that gap is the work nanoGPT has already done.

## What "training on real data" actually requires

Concretely: imagine pointing `tiny_decoder.py` at a 32 MB text file (Tiny Shakespeare scaled up ~30×, or about 8M BPE tokens). Here's what happens, stage by stage:

| Stage | What breaks in tiny_decoder today | What it actually needs |
|---|---|---|
| **Tokenize the corpus** | `TinyTokenizer.encode` raises `KeyError` on the first unseen word. The vocabulary is closed and word-level; there's no `<unk>`, no BPE, no fallback. | Replace with `tiktoken` (GPT-2 BPE, ~50K vocab) or a SentencePiece model trained on the corpus. Touches the entire pipeline. |
| **Build the dataset** | `build_dataset` materialises every (input, target) pair into a single tensor in memory. At max_len=16 and 8M tokens that's ~500K rows just for inputs. `train()` then runs `model(inputs)` over the *entire dataset* in one forward pass — activations blow past any consumer GPU's VRAM long before the backward pass. | Rewrite as a `DataLoader` reading random windows from an `np.memmap`'d binary file. nanoGPT's `get_batch` is the reference (~15 lines). |
| **Run the training loop** | Constant LR=3e-3, no warmup, no decay, no gradient accumulation, no mixed precision, no checkpointing, no validation split. Even with mini-batches added, the model won't converge cleanly and you have no way to monitor or recover from failure. | Linear warmup → cosine decay, bf16/fp16 autocast, grad accumulation, periodic checkpointing, held-out val loss. ~150 lines of training engineering. |
| **Generate samples** | No KV cache — generation is O(T²). At max_len=512 that's 250K wasted forward passes per sequence. Tolerable but slow. | KV cache: ~50 lines plus an attention signature change (`forward(x, past_kv=None) -> (out, new_kv)`). |
| **Get useful output** | Default model is `n_layers=2, d_model=64`, ~30K non-embedding parameters. With a 50K BPE vocab, the embedding table (3.2M params) dwarfs the model by 100×. There is no actual language-modelling capacity. | Scale to `n_layers=6, d_model=384`, ~10M params — config change only, no code. But at this point the residual-init scaling, the multi-head attention, and the absolute-positional limitations all start mattering for real. |

Total cost: roughly **350 lines of new training infrastructure, a tokenizer rewrite, and at least one new dependency.** The file roughly doubles in size, and the part of it that's about *architecture* — the part that gave the file its reason to exist — becomes a third of the content instead of nearly all of it.

## The "poor implementation of nanoGPT" argument

Every piece of code in the table above is code that nanoGPT already contains, in working form, in ~600 lines. Karpathy's design choices for `train.py`, `model.py`, and the prep scripts are the same ones any thoughtful re-implementation would arrive at:

- `tiktoken` for tokenization
- `np.memmap` over a binary `train.bin` / `val.bin` for streaming the corpus
- `get_batch(split)` sampling random windows
- Linear warmup followed by cosine decay
- `torch.amp.autocast` + `GradScaler` for mixed precision
- `gradient_accumulation_steps` to decouple effective batch size from memory
- Periodic eval against val.bin
- Resumable checkpoints

This isn't because there's something special about nanoGPT — it's because there's a narrow set of correct answers for each of these problems, and nanoGPT picked them. Any version you build inside tiny_decoder would either converge on the same choices (in which case you've re-written nanoGPT, less well-tested) or diverge from them (in which case you've made the file worse, not better).

The architectures are already the same — the [refactor](https://github.com/teleoric/tiny-transformer/pull/1) made it so deliberately. Pre-norm blocks, GELU FFN, tied embeddings, GPT-2 residual init scaling, parameter-group weight decay, shifted-target training, shared causal mask, AdamW. The model code in nanoGPT and tiny_decoder is line-for-line equivalent on the architecture; what nanoGPT adds is *the engineering around it*.

So the structural claim is this: **augmenting tiny_decoder to train on real data isn't extending it — it's reimplementing nanoGPT inside it.** That trade is bad on three axes:

1. **You lose tiny_decoder's stated virtue.** "Read every line in 15 minutes" stops being true once the architecture is a third of a 2000-line file.
2. **You're maintaining a worse nanoGPT.** Less battle-tested, less performance-tuned, fewer people debugging it. Karpathy's repo has ~40K GitHub stars and has been training real models for years; your fork has neither.
3. **You haven't moved the unique work forward.** The 350 lines of training infrastructure aren't where your team's competitive advantage lives. They're commodity engineering that's already been written.

## What to do instead

> Read tiny_decoder until you understand every line of the architecture. Then clone nanoGPT, run `python data/shakespeare_char/prepare.py && python train.py config/train_shakespeare_char.py` to verify your setup works end-to-end in ~15 minutes on a single GPU, then point its data prep script at your real corpus and bump the model config. tiny_decoder is the explainer; nanoGPT is the trainer. They share an architecture by design.

If for organisational reasons keeping the work in this repo matters, the honest path is to **vendor nanoGPT alongside `tiny_decoder.py`** — add it as a sibling top-level file (`nano_decoder.py` or a subdirectory) and let `tiny_decoder.py` stay frozen as the architecture reference. Don't grow tiny_decoder into nanoGPT one PR at a time; you'll end up with a worse nanoGPT *and* a worse tiny_decoder.

The diagnostic question worth asking before starting any of this: **if a perfectly-functioning training pipeline appeared in this repo tomorrow, would the team be relieved or disappointed?** If disappointed, the work *is* the point and rebuilding from scratch is a legitimate eyes-open choice. If relieved, you're paying NIH tax — and the cheaper path is the link above.

---

# Recommended next steps

Three recommendations, each at a different point on the simplicity-to-production spectrum:

### 1. nanoGPT — Best for Learning

Andrej Karpathy's implementation. Single-file GPT-2 training that's readable end-to-end. Your `tiny_decoder.py` is now structurally aligned with nanoGPT — same pre-norm blocks, GELU, tied embeddings, GPT-2 residual init scaling, parameter-group weight decay, shifted-target training, and shared causal mask. Stepping up to nanoGPT primarily means: a real BPE tokenizer (tiktoken), a KV cache for fast generation, mini-batched training with a DataLoader, mixed precision, and a non-trivial dataset.

- ~300 lines of core model code
- Trains GPT-2 124M on OpenWebText, scales to 350M+
- Pure PyTorch, no framework abstractions
- Works on ROCm — it's just `torch.nn` modules
- DDP support for multi-GPU (if you ever add a second card)

```bash
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
python train.py config/train_gpt2.py
```

For 300M on 24GB VRAM: you'll need gradient accumulation and possibly activation checkpointing. The codebase is simple enough to add both yourself, which is the learning value.

### 2. torchtune — Best for Practical Fine-Tuning on ROCm

PyTorch-native, maintained by the PyTorch team. No dependency on bitsandbytes or CUDA-specific kernels — it's pure PyTorch ops, so ROCm works cleanly.

- Recipes for full fine-tune, LoRA, QLoRA
- Supports Llama, Mistral, Gemma, Phi architectures
- Built-in configs for different GPU memory budgets
- Composable — each component (model, tokenizer, recipe) is standalone and inspectable

```bash
pip install torchtune
tune download meta-llama/Llama-3.1-8B-Instruct
tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device
```

Less useful for training from scratch at 300M, but excellent for understanding how production fine-tuning pipelines work — and immediately applicable to your RAG use case if you need to specialize model behavior.

### 3. GPT-NeoX / LitGPT — Best for Scaling to 300M and Beyond

**LitGPT** (Lightning AI) is the more actively maintained option:

- Supports 20+ architectures (Llama, Mistral, Phi, StableLM, Pythia, etc.)
- Training from scratch + fine-tuning + inference in one codebase
- Explicit small-model configs (Pythia 70M/160M/410M) — perfect for the 300M target
- ROCm compatible — pure PyTorch with optional FSDP

```bash
pip install litgpt
litgpt pretrain --config pythia-160m  # start small
# scale to custom 300M config
```

**Pythia** (EleutherAI) model configs are particularly good for learning — the training data, checkpoints at every stage, and model code are all open. You can reproduce or extend training from any checkpoint.

---

### What I'd Actually Do

Start with **nanoGPT** — read it, understand every line, modify it. It's the closest thing to a production-readable `tiny_decoder.py` and the diff between the two is largely about scale, tokenization, and engineering polish (KV cache, mini-batching, mixed precision) rather than architecture. Train a 124M GPT-2 on your 7900 XT to validate the full pipeline works on ROCm. Then scale to 300M by adjusting `n_layer`, `n_head`, `n_embd` in the config.

Once you've internalized the training mechanics, move to **torchtune** for fine-tuning real models (Llama 8B with LoRA) — that's where the practical value lands for your RAG platform.

| | nanoGPT | torchtune | LitGPT |
|---|---|---|---|
| Learning value | Highest | Medium | Medium |
| Code readability | ~300 lines | Modular, more files | Large codebase |
| Train from scratch | Yes | No (fine-tune only) | Yes |
| Fine-tune existing | Manual | Built-in recipes | Built-in |
| 300M from scratch | With tweaks | Not designed for it | Out of the box |
| ROCm compatibility | Works (pure PyTorch) | Works (pure PyTorch) | Works (pure PyTorch) |
| Production relevance | Educational | High | High |
