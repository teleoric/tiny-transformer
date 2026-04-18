## Glossary

### Foundations (tiny_decoder.py)

**Token** — The smallest unit of text the model processes. In `tiny_decoder.py`, tokens are whole words (`"cats"`, `"like"`). Production tokenizers (BPE, SentencePiece) split words into subword pieces — `"accounting"` might become `["account", "ing"]`.

**Vocabulary** — The fixed set of all tokens the model knows. `tiny_decoder.py` has 8 tokens. Llama-3.1 has 128K. Tokens outside the vocabulary cannot be processed.

**Token Embedding** — A learned vector representation for each token in the vocabulary. `nn.Embedding` maps integer token IDs to dense vectors of dimension `d_model`. The model learns these vectors during training — semantically similar tokens end up with similar vectors.

**Positional Embedding** — A learned vector added to each token embedding to encode its position in the sequence. Without this, the model can't distinguish `"I like cats"` from `"cats like I"` because attention is permutation-invariant. `tiny_decoder.py` uses learned positional embeddings; the original Transformer used sinusoidal functions; Llama uses RoPE (rotary position embeddings).

**d_model** — The hidden dimension of the model. Every token flows through the network as a vector of this size. `tiny_decoder.py` uses 32; Llama-8B uses 4096. Wider models can represent more nuanced relationships but use more memory.

**d_ff** — The inner dimension of the feed-forward network. Typically 2-4x `d_model`. This is where the model does most of its "thinking" — pattern matching, factual recall, and transformation happen in this expanded space.

**Attention** — The mechanism that lets each token look at other tokens in the sequence to build context-dependent representations. Computes a weighted sum of all token representations, where the weights (attention scores) are learned based on query-key similarity.

**Query / Key / Value (Q, K, V)** — Three linear projections of each token's representation. Queries ask "what am I looking for?", keys answer "what do I contain?", and the dot product between them determines how much each token attends to every other. Values carry the actual information that gets aggregated.

**Scaled Dot-Product Attention** — The specific attention formula: `softmax(QK^T / sqrt(d_k)) × V`. The scaling by `sqrt(d_k)` prevents the dot products from growing too large, which would push softmax into regions with vanishingly small gradients.

**Causal Mask** — A lower-triangular boolean matrix that prevents each token from attending to future tokens. This is what makes the model autoregressive — token 3 can see tokens 0, 1, 2 but not token 4. Implemented in `tiny_decoder.py` as `torch.tril()`. Without this, the model would "cheat" during training by seeing the answer.

**Multi-Head Attention** — Running multiple attention operations in parallel, each with its own Q/K/V projections, on slices of `d_model`. Each head can learn different relationship types (syntactic, semantic, positional). `tiny_decoder.py` uses 1 head; Llama-8B uses 32.

**Feed-Forward Network (FFN)** — A two-layer MLP applied independently to each position after attention. `d_model → d_ff → d_model` with a nonlinearity in between. `tiny_decoder.py` uses ReLU; Llama uses SwiGLU.

**Residual Connection** — Adding the input of a sub-layer to its output: `x + sublayer(x)`. Prevents gradient degradation in deep networks and lets the model learn incremental refinements rather than full transformations at each layer.

**Layer Normalization (LayerNorm)** — Normalizes activations across the feature dimension to stabilize training. Pre-norm (applied before attention/FFN) is used in `tiny_decoder.py` and all modern models. Post-norm (applied after) was used in the original Transformer.

**Transformer Block** — One complete unit: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual. `tiny_decoder.py` has 1 block. GPT-2 has 12-48. Llama-8B has 32. Stacking blocks gives the model depth to learn increasingly abstract representations.

**Logits** — Raw unnormalized scores output by the model's final linear layer, one per token in the vocabulary. The highest logit corresponds to the model's best guess for the next token. Softmax converts logits to probabilities.

**Cross-Entropy Loss** — The training objective. Measures how far the model's predicted probability distribution is from the true next token. Lower is better. A perfect model assigns probability 1.0 to the correct next token.

**Autoregressive Generation** — Producing text one token at a time, feeding each predicted token back as input for the next step. The `generate()` function in `tiny_decoder.py` does this in a loop until `<eos>` or `max_len`. This is inherently sequential — each token depends on all previous tokens.

**Left Padding** — Prepending `<pad>` tokens so all sequences are the same length. `tiny_decoder.py` left-pads because the model reads the last position for its prediction. Right-padding is used in encoder models and batch training. The distinction matters for causal models.

**Special Tokens** — Tokens with structural meaning rather than linguistic content. `<bos>` (beginning of sequence) signals the start; `<eos>` (end of sequence) signals the model should stop generating; `<pad>` fills unused positions and is masked out.

**Overfitting** — When the model memorizes the training data perfectly but can't generalize to new inputs. `tiny_decoder.py` intentionally overfits — it has 9K parameters learning 8 examples. This is a feature for demonstration, not a flaw.

**Argmax Decoding** — Selecting the token with the highest logit at each step. Deterministic — always produces the same output. `tiny_decoder.py` uses this. Production systems use sampling with temperature, top-k, or top-p for variety.

---

### Training & Open Source Models (nanoGPT, torchtune, LitGPT)

**Pre-Training** — Training a model from scratch on a large corpus (billions of tokens) to learn general language understanding. This is what nanoGPT does with OpenWebText. Extremely compute-intensive — weeks on multi-GPU clusters.

**Fine-Tuning** — Taking a pre-trained model and further training it on a smaller, task-specific dataset. Adjusts the model's behavior without starting from scratch. Much cheaper than pre-training.

**Full Fine-Tune** — Updating every parameter in the model during fine-tuning. Produces the best quality adaptation but requires memory for all weights + optimizer states + gradients. An 8B model needs ~60GB+ for full fine-tuning.

**LoRA (Low-Rank Adaptation)** — Fine-tuning technique that freezes the base model and injects small trainable matrices into attention layers. Instead of updating a `4096×4096` weight matrix, you train two small matrices `4096×r` and `r×4096` where `r` (rank) is typically 8-64. Reduces trainable parameters by 100-1000x.

**QLoRA (Quantized LoRA)** — LoRA with the base model loaded in 4-bit precision (NF4). Cuts weight memory by 4x, enabling fine-tuning of 8B models on 24GB consumer GPUs. The LoRA adapters themselves remain in FP16 for gradient stability.

**LoRA Rank** — The `r` parameter in LoRA. Higher rank = more trainable parameters = more expressive adaptation but more memory. Rank 8-16 is typical for most tasks; rank 64+ for complex domain adaptation.

**LoRA Adapter** — The small set of trained weight matrices produced by LoRA fine-tuning. Typically ~50-200MB for an 8B model. Can be merged into the base model or loaded dynamically at serving time.

**Gradient Accumulation** — Simulating a larger batch size by accumulating gradients across multiple forward/backward passes before updating weights. Trades compute time for memory — essential when a single batch doesn't fit in VRAM.

**Activation Checkpointing (Gradient Checkpointing)** — Discarding intermediate activations during the forward pass and recomputing them during backpropagation. Reduces memory at the cost of ~30% more compute. Critical for training large models on limited VRAM.

**Mixed Precision Training** — Using FP16 or BF16 for forward/backward passes while keeping a master copy of weights in FP32 for numerical stability. Halves memory for activations and doubles throughput on hardware with tensor cores.

**BF16 (bfloat16)** — A 16-bit floating point format with the same exponent range as FP32 but reduced mantissa precision. Better for training than FP16 because it doesn't overflow on large gradients. Native on A100/H100; supported on RDNA3 via ROCm.

**NF4 (NormalFloat 4-bit)** — A 4-bit quantization format designed for normally distributed weights. Used by QLoRA. Each weight is mapped to one of 16 levels optimized for Gaussian distributions, with a per-block scaling factor.

**Optimizer States** — Additional per-parameter data maintained by the optimizer. Adam stores two states per parameter (first and second moment estimates), tripling the memory footprint beyond the weights themselves. This is why full fine-tuning is so expensive.

**Batch Size** — The number of training examples processed together in one forward/backward pass. Larger batches give more stable gradients but use more memory. `tiny_decoder.py` uses full-batch (all 8 examples at once).

**Mini-Batching** — Processing the training data in small chunks (mini-batches) instead of all at once. Enables training on datasets larger than GPU memory. Standard practice for any real-world training.

**DataLoader** — PyTorch utility that handles batching, shuffling, and feeding data to the training loop. Absent from `tiny_decoder.py` (full-batch), required for any real training pipeline.

**Epoch** — One complete pass through the entire training dataset. `tiny_decoder.py` trains for 600 epochs. Pre-training large models typically runs for 1-2 epochs over the full corpus; fine-tuning runs 1-5 epochs.

**Learning Rate** — How much to adjust weights in response to each gradient. Too high → training diverges. Too low → training stalls. `tiny_decoder.py` uses 3e-2 (high, works because the model is tiny). Production training uses 1e-4 to 5e-5 with warmup and decay schedules.

**Warmup** — Gradually increasing the learning rate from near-zero to the target value over the first N steps. Prevents early instability when the model hasn't yet found a reasonable region of the loss landscape.

**Perplexity** — A measure of how surprised the model is by the test data. Defined as `exp(cross_entropy_loss)`. Lower is better. A perplexity of 10 means the model is, on average, as uncertain as if it were choosing between 10 equally likely tokens.

**Checkpoint** — A saved snapshot of model weights and optimizer state at a point during training. Enables resuming interrupted training and selecting the best-performing version of the model.

**SwiGLU** — The activation function used in Llama and most modern LLMs. Replaces ReLU in the FFN with a gated variant: `SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)`. Empirically better than ReLU but adds a third weight matrix to the FFN.

**RoPE (Rotary Position Embeddings)** — Encodes position information by rotating Q and K vectors using sinusoidal functions based on position. Unlike learned positional embeddings, RoPE naturally generalizes to longer sequences than seen during training.

**GQA (Grouped Query Attention)** — A variant where multiple query heads share a single key/value head. Reduces KV cache size during inference without significantly impacting quality. Used by Llama 3.1.

**Tokenizer (BPE/SentencePiece)** — Production tokenizers that split text into subword units via byte-pair encoding or similar algorithms. Handle any input (no unknown tokens) by decomposing rare words into character-level or byte-level pieces. Unlike `tiny_decoder.py`'s word-level tokenizer.

---

### Inference & Serving (vLLM)

**Inference** — Running a trained model to produce outputs, without updating weights. The forward pass only — no backpropagation, no gradients, no optimizer.

**Prefill** — The first phase of inference where the model processes the entire input prompt in parallel. Compute-bound — the GPU's FLOPS are the bottleneck. Produces the KV cache for all input tokens.

**Decode** — The second phase where the model generates tokens one at a time. Memory-bandwidth-bound — each step reads the full KV cache but produces only one token. This is why inference is slower per-token than prefill.

**KV Cache** — Stored key and value tensors from previous tokens' attention computations. Without caching, generating token N would require recomputing attention for all N-1 previous tokens from scratch. `tiny_decoder.py`'s `generate()` recomputes everything each step — the KV cache is what eliminates this redundancy at scale.

**Paged Attention** — vLLM's memory management technique inspired by OS virtual memory. KV cache is stored in non-contiguous pages rather than one contiguous block per sequence. Eliminates memory fragmentation and enables efficient dynamic allocation as sequences grow and shrink.

**Block Size** — The number of tokens per page in paged attention. vLLM defaults to 16. Smaller blocks reduce internal fragmentation but increase management overhead.

**Continuous Batching** — Processing multiple requests simultaneously, adding new requests as old ones finish. Unlike static batching (wait for all to finish), continuous batching keeps the GPU saturated. vLLM does this automatically.

**Chunked Prefill** — Breaking long prompts into chunks and interleaving prefill computation with decode steps from other requests. Prevents a single long prompt from blocking all decode operations.

**Throughput** — Tokens per second across all concurrent requests. The key server-side metric. vLLM on RDNA3 with Llama-8B achieves ~46.6 tok/s output throughput.

**TTFT (Time to First Token)** — Latency from receiving the request to producing the first output token. Dominated by prefill time. The key user-facing latency metric.

**TPOT (Time per Output Token)** — Average time between successive output tokens after the first. Dominated by decode speed and KV cache reads.

**ITL (Inter-Token Latency)** — Time between each consecutive token. Similar to TPOT but measured per-token rather than averaged. Captures variance — a user notices inconsistent token delivery as "stuttering."

**Quantization (Inference)** — Reducing weight precision to decrease memory usage and increase throughput. AWQ and GPTQ use 4-bit integer representations. FP8 uses 8-bit floating point (not available on RDNA3). Unlike training quantization (NF4/QLoRA), inference quantization is applied to frozen weights.

**AWQ (Activation-Aware Weight Quantization)** — A 4-bit quantization method that identifies and preserves high-importance weights based on activation patterns. Generally higher quality than GPTQ for the same bit width.

**GPTQ** — A 4-bit weight quantization method using approximate second-order information. Fast to apply, well-supported across inference engines. Slightly lower quality than AWQ on some benchmarks.

**Enforce Eager** — Disabling CUDA graph capture and `torch.compile`, running operations one at a time. More debuggable and stable on RDNA3, but slower than compiled execution.

**CUDA Graphs** — Pre-recording a sequence of GPU operations and replaying them without CPU involvement. Eliminates kernel launch overhead. Stable on CDNA/NVIDIA; may hang on RDNA3 during warmup.

**TunableOp / GEMM Tuning** — PyTorch's mechanism for benchmarking all available GPU kernel implementations for each matrix multiplication shape and caching the fastest. Critical on RDNA3 because hipBLAS (the fallback GEMM library) has suboptimal default kernel selection.

**Prefix Caching** — Reusing KV cache blocks across requests that share the same prefix (typically system prompts). If 100 requests share a 500-token system prompt, the KV cache for those 500 tokens is computed once and shared.

**Swap Space** — CPU memory reserved for KV cache blocks that are evicted from GPU when VRAM is full. Preempted sequences can be resumed by swapping blocks back to GPU rather than recomputing from scratch.

**gpu_memory_utilization** — vLLM parameter controlling what fraction of GPU VRAM is pre-allocated for KV cache (after model weights). 0.92 means "use 92% of total VRAM." Higher values leave more room for KV cache but less margin for memory spikes.

---

### RAG & Distillation (Client Architecture)

**RAG (Retrieval-Augmented Generation)** — An architecture that retrieves relevant documents from a knowledge base and injects them into the LLM's prompt as context. The model generates answers grounded in the retrieved content rather than relying solely on its parametric memory. Separates "what the model knows" (training) from "what it can access" (retrieval).

**Embedding Model** — A model that converts text into dense vector representations where semantic similarity maps to vector proximity. Used to encode both documents (at indexing time) and queries (at search time). Common choices: BGE, E5, GTE. Runs on CPU or GPU independently of the inference LLM.

**Vector Store** — A database optimized for storing and searching dense vector embeddings. Qdrant, Pinecone, Weaviate, Milvus are common options. Supports approximate nearest-neighbor search to find the most semantically similar documents to a query.

**Chunking** — Splitting source documents into smaller segments for embedding and retrieval. Chunk size affects retrieval quality — too large and irrelevant content dilutes the signal; too small and context is fragmented. Typical sizes: 256-1024 tokens with overlap.

**Semantic Search** — Finding documents by meaning rather than keyword match. The query and documents are embedded into the same vector space, and the nearest vectors are returned. Enables finding relevant content even when the exact words differ.

**Context Window** — The maximum number of tokens the model can process in a single forward pass. GPT-5.4 supports 1.05M tokens. Llama-3.1-8B supports 128K. Retrieved documents must fit within this window alongside the system prompt and query.

**System Prompt** — Instructions prepended to every conversation that define the model's behavior, output format, and constraints. In the accounting context: citation formatting rules, GAAP/IFRS terminology preferences, audit workpaper templates. Not trained into the model — provided at inference time.

**Citation Provenance** — Tracing each claim in the model's output back to a specific retrieved document and passage. Critical for audit and legal contexts where every statement must be verifiable. Implemented via structured output (tool use) or post-processing.

**Grounding** — Constraining the model's output to information present in the retrieved context. A well-grounded model won't hallucinate facts that aren't in the provided documents. Achieved through prompt engineering, fine-tuning, or both.

**Hallucination** — When the model generates plausible-sounding but factually incorrect content not supported by the retrieved context or reality. The primary risk in professional RAG deployments. Fine-tuning can reduce hallucination rate; retrieval quality is the other lever.

**Distillation** — Using a large, expensive model (the teacher) to generate high-quality outputs, then training a smaller, cheaper model (the student) to replicate those outputs. The student learns to mimic the teacher's behavior on specific tasks without needing the teacher's full capability.

**Teacher Model** — The large model whose outputs serve as training signal for distillation. In the client scenario, GPT-5.4 is the teacher. It produces the "gold standard" responses that the student model learns from.

**Student Model** — The smaller model being trained via distillation. Typically 5-20x smaller than the teacher. Targets the same quality on the specific task at much lower inference cost. In OpenAI's ecosystem: gpt-4.1-mini or gpt-4.1-nano. In open source: Llama-8B or Mistral-7B.

**Stored Completions** — OpenAI's mechanism for automatically capturing input-output pairs from API calls. Setting `store: true` on API requests saves them for later use as distillation training data. Retained for 30 days by default.

**Supervised Fine-Tuning (SFT)** — Training a model on (input, desired_output) pairs. The most straightforward fine-tuning method. The distilled dataset from the teacher model becomes the SFT training data for the student.

**Reinforcement Fine-Tuning (RFT)** — Training a model using a reward signal (grader) rather than explicit output examples. The model generates candidate outputs and learns from which ones score highest. Useful when the desired behavior is easier to evaluate than to demonstrate.

**Direct Preference Optimization (DPO)** — Training a model on pairs of (preferred output, rejected output) for the same input. Teaches the model which style of response is better without needing a separate reward model. Used to align model behavior with human preferences.

**Eval (Evaluation)** — Systematic measurement of model output quality against defined criteria. Before fine-tuning: establishes baseline. After fine-tuning: measures improvement. Without evals, you're tuning blind. OpenAI provides platform evals; open-source alternatives include lm-eval-harness.

**Data Residency** — The requirement that data remain within specific geographic or network boundaries. Accounting firms handling audit workpapers may require that client data never leaves their on-prem infrastructure, which rules out cloud-hosted API calls to OpenAI.

**On-Prem Inference** — Running model inference on hardware you physically control rather than calling a cloud API. Eliminates data egress concerns, removes per-token costs, but requires infrastructure investment and operational expertise. The H100s in the scenario serve this purpose.


