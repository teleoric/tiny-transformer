Three recommendations, each at a different point on the simplicity-to-production spectrum:

### 1. nanoGPT — Best for Learning

Andrej Karpathy's implementation. Single-file GPT-2 training that's readable end-to-end. Your `tiny_decoder.py` is structurally very close to this — stepping up to nanoGPT is a natural progression.

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

Start with **nanoGPT** — read it, understand every line, modify it. It's the closest thing to a production-readable `tiny_decoder.py`. Train a 124M GPT-2 on your 7900 XT to validate the full pipeline works on ROCm. Then scale to 300M by adjusting `n_layer`, `n_head`, `n_embd` in the config.

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


