# Research Plan: Recursive Multiplication via Looped Transformers

## Thesis

Current approaches to teaching transformers multiplication use scratchpads that mirror the **school (grade-school / long) multiplication algorithm**, which has O(n²) computational complexity and requires each output digit to attend to a linearly growing number of input digits. This makes length generalisation inherently difficult for fixed-depth architectures.

**Core hypothesis:** A looped transformer trained on a **Karatsuba-style recursive decomposition** of multiplication will achieve significantly better length generalisation than the same architecture trained on a flat/school-algorithm decomposition, because the per-step computation is bounded regardless of input length, and the recursion depth grows only as log(n).

No prior work (as of March 2026) has explored teaching transformers Karatsuba-style recursive multiplication. The closest results are:

- **Cho et al. (ICLR 2025)** — achieved ~2-3× length generalisation on multiplication using task-specific scratchpads + multi-level position coupling, but their scratchpad mirrors long multiplication, not a recursive algorithm
- **Hou, Brandfonbrener et al. (ICML 2025)** — "Universal Length Generalization with Turing Programs." Achieves 97%+ accuracy on 100-digit multiplication (trained on 50-digit) by decomposing tasks into Turing Machine-style steps with Hard-ALiBi. First strong multiplication length generalisation result, but uses a flat TM simulation rather than recursive decomposition. This is the new SOTA baseline to beat.
- **Fan et al. (ICLR 2025)** — showed looped transformers with adaptive depth significantly improve length generalisation on iterative tasks (addition, parity), but did not attempt multiplication
- **Jelassi et al. (2023)** — showed train-set priming helps generalise on N×3 multiplication (one operand fixed), but used flat decomposition

This plan combines the looped architecture from Fan et al. with a Karatsuba-structured scratchpad and hierarchical position encodings, targeting the open problem of N×N multiplication length generalisation.

**Key differentiator vs. Hou et al. (ICML 2025):** Their Turing Programs approach works but uses O(n²) steps for multiplication (simulating a TM). Our recursive decomposition should achieve the same generalisation with O(n^1.585) work and O(log n) recursion depth, which is both theoretically cleaner and more computationally efficient at scale.

---

## Background You Need to Understand

### The Karatsuba Algorithm

To multiply two n-digit numbers X and Y:

1. **Split** each number in half: X = X_high · B^(n/2) + X_low, Y = Y_high · B^(n/2) + Y_low (where B is the base)
2. **Compute three sub-products:**
   - z0 = X_low × Y_low
   - z2 = X_high × Y_high
   - z1 = (X_low + X_high) × (Y_low + Y_high) − z0 − z2
3. **Combine:** X × Y = z2 · B^n + z1 · B^(n/2) + z0

The key insight is that step 2 requires only **3 multiplications of half-sized numbers** (not 4), giving O(n^1.585) complexity. Recursion bottoms out when numbers are small enough to multiply directly (e.g., single-digit × single-digit).

**Why this matters for transformers:** Each recursive step involves only bounded-complexity operations (splitting, addition, subtraction, shifting). The multiplications at each level are on numbers half the size. A looped transformer can implement one level of recursion per loop iteration, with log₂(n) total iterations.

### Key Papers to Read

**On position coupling (the foundation you're building on):**
- Cho et al. (NeurIPS 2024) — "Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure." Introduces the idea of assigning the same position ID to tokens that are "structurally relevant" to each other (e.g., digits of the same significance). Demonstrated 6.67× generalisation on addition. Applied to N×2 multiplication but not general N×N. Code at github.com/HanseulJo/position-coupling.
- Cho et al. (ICLR 2025) — "Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count." Extends position coupling to multiplication with scratchpads achieving ~2-3× generalisation.
- Shen et al. (EMNLP 2025) — "Positional Description Matters for Transformers Arithmetic." Modifying positional encoding or task representation; trained 100M param model achieving strong 15-digit multiplication. Relevant for understanding how position information helps arithmetic. [arxiv.org/abs/2311.14737](https://arxiv.org/abs/2311.14737)

**On looped transformers (the architecture you'll use):**
- Fan et al. (ICLR 2025) — "Looped Transformers for Length Generalization." Shows that weight-shared transformers with adaptive loop count generalise much better on iterative tasks. Proposes a training algorithm for looped transformers on RASP-L expressible tasks. This is the architecture backbone.
- Saunshi et al. (ICLR 2025) — "Reasoning with Latent Thoughts: On the Power of Looped Transformers." **Critical new paper.** Proves a k-layer transformer looped L times nearly matches a kL-layer non-looped model. Shows looped models implicitly generate "latent thoughts" and can simulate T steps of CoT with T loops. Proposes looping-based regularisation. Directly validates our approach of using loops for recursive computation. [arxiv.org/abs/2502.17416](https://arxiv.org/abs/2502.17416)
- **LoopFormer — Jeddi et al. (Feb 2026)** — "Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation." Introduces shortcut-consistency training that aligns trajectories of different loop lengths. Conditions each loop step on internal time t and step size dt, allowing coarser schedules to approximate fine-grained ones. **Directly relevant to our adaptive loop count problem.** [arxiv.org/abs/2602.11451](https://arxiv.org/abs/2602.11451)
- **SpiralFormer — Yu, Shu et al. (Feb 2026)** — "Looped Transformers Can Learn Hierarchical Dependencies via Multi-Resolution Recursion." Executes recurrence under a multi-resolution schedule: early iterations capture global interactions on compressed sequences, later iterations refine at token resolution. **Very relevant — their multi-resolution approach mirrors our hierarchical recursion levels.** [arxiv.org/abs/2602.11698](https://arxiv.org/abs/2602.11698)
- **Parallel Loop Transformer — Wu et al. (Oct 2025)** — Uses Cross-Loop Parallelism (CLP) to break sequential dependency in looped transformers. Could be useful for speeding up inference at test time on longer sequences. [arxiv.org/abs/2510.24824](https://arxiv.org/abs/2510.24824)
- Relaxed Recursive Transformers (ICLR 2025, Google DeepMind) — Efficiently initialises recursive transformers from pretrained models using per-layer LoRA. Proposes Continuous Depth-wise Batching for 2-3× inference throughput. [arxiv.org/abs/2410.20672](https://arxiv.org/abs/2410.20672)
- Giannou et al. (ICML 2023) — "Looped Transformers as Programmable Computers." Proves looped transformers can emulate basic computational primitives. Gives the theoretical foundation for why this approach should work.
- Dehghani et al. (2018) — "Universal Transformers." The original weight-sharing / looped transformer paper. Introduced adaptive computation time (ACT) for deciding when to stop iterating.

**On multiplication length generalisation (the problem you're attacking):**
- **Hou, Brandfonbrener et al. (ICML 2025)** — "Universal Length Generalization with Turing Programs." **New SOTA.** 98% accuracy on 100-digit addition (trained on 50-digit), 97%+ on 100-digit multiplication. Uses Hard-ALiBi attention bias. First strong multiplication result. **Must-read and primary comparison target.** [arxiv.org/abs/2407.03310](https://arxiv.org/abs/2407.03310)
- **Cai et al. (NeurIPS 2025)** — "Extrapolation by Association: Length Generalization Transfer in Transformers." Shows length generalisation can transfer across related tasks — training with a longer auxiliary task enables generalisation on a shorter target task. **Potential technique:** train on longer addition (easy to generalise) to bootstrap multiplication generalisation. [arxiv.org/abs/2506.09251](https://arxiv.org/abs/2506.09251)
- McLeish et al. (NeurIPS 2024) — "Transformers Can Do Arithmetic with the Right Embeddings." Per-digit position embeddings → 99% accuracy on 100-digit addition trained on 20-digit. Also improves multiplication. [arxiv.org/abs/2405.17399](https://arxiv.org/abs/2405.17399)
- Jelassi et al. (2023) — "Length Generalization in Arithmetic Transformers." Showed relative position embeddings fail for multiplication; proposed train-set priming. Got 5-digit × 3-digit → 35 × 3 generalisation, but only for fixed second operand length.
- Duan et al. (ICLR 2024) — "From Interpolation to Extrapolation: Complete Length Generalization for Arithmetic Transformers." Introduced Attention Bias Calibration (ABC).
- Back de Luca et al. (NeurIPS 2025) — "Learning to Add, Multiply, and Execute Algorithmic Instructions Exactly with Neural Networks." NTK framework proves exact arithmetic is learnable from logarithmically many samples. Theoretical backing for our approach. [arxiv.org/abs/2502.16763](https://arxiv.org/abs/2502.16763)

**On length generalisation theory (new 2025-2026 results):**
- **Izzo et al. (Oct 2025)** — "Quantitative Bounds for Length Generalization in Transformers." First quantitative bounds on required training length. Proves length generalisation occurs when internal behaviour on longer sequences can be "simulated" by shorter ones — directly supports our recursive decomposition (each level reduces to shorter sub-problems). [arxiv.org/abs/2510.27015](https://arxiv.org/abs/2510.27015)
- **(Feb 2026)** — "Length Generalization Bounds for Transformers." Proves non-existence of computable length generalisation bounds for CRASP (2-layer), but provides computable bounds for the positive fragment. Landmark negative result — important for understanding theoretical limits. [arxiv.org/abs/2603.02238](https://arxiv.org/abs/2603.02238)

**On scratchpad / chain-of-thought ordering:**
- **Sato et al. (ICML 2025 Workshop)** — "Chain of Thought in Order: Discovering Learning-Friendly Orders for Arithmetic." Order of reasoning steps critically affects difficulty. Proposes a hierarchical approach to discover optimal orderings. On multiplication, recovers the reverse-digit order. **Relevant to our scratchpad format decision** — the order in which we present the Karatsuba trace matters. [arxiv.org/abs/2506.23875](https://arxiv.org/abs/2506.23875)
- Coconut — Hao et al. (ICLR 2025) — "Training LLMs to Reason in a Continuous Latent Space." Uses hidden states as "continuous thought" fed back as input, enabling BFS-like reasoning. **Connection to looped transformers:** our loops are essentially doing continuous latent reasoning without explicit token generation at each step. [arxiv.org/abs/2412.06769](https://arxiv.org/abs/2412.06769)

**On mechanistic interpretability of arithmetic (for understanding what the model learns):**
- Nanda et al. (ICLR 2023) — "Progress Measures for Grokking via Mechanistic Interpretability." Fully reverse-engineered a transformer doing modular addition: it uses discrete Fourier transforms and trigonometric identities. Key insight: the model doesn't learn "addition rules" — it learns to embed numbers onto rotations in R² and compose them. Essential reading for understanding what representations your model might discover.

**On neural algorithmic reasoning (the broader context):**
- Veličković & Blundell (2021) — "Neural Algorithmic Reasoning." The survey/manifesto for the field.
- **Wittig et al. (Feb 2026)** — "Which Algorithms Can Graph Neural Networks Learn?" Theoretical framework for when MPNNs can learn algorithms. Establishes impossibility results too. Relevant for understanding what architectural constraints enable algorithmic learning. [arxiv.org/abs/2602.13106](https://arxiv.org/abs/2602.13106)

**On position encoding theory:**
- **(June 2025)** — "Theoretical Analysis of Positional Encodings in Transformer Models." Proposes novel encodings based on orthogonal function families (wavelets, Legendre polynomials). Could inspire alternatives to our hierarchical position scheme. [arxiv.org/abs/2506.06398](https://arxiv.org/abs/2506.06398)

---

## Architecture Decisions

### Decision 1: Number Representation

**Options to explore (ranked by priority):**

1. **Binary representation** (RECOMMENDED START)
   - Karatsuba is cleanest in binary: splitting is just taking the top/bottom half of bits
   - Each token = one bit (0 or 1)
   - Position encoding marks bit significance (ones, twos, fours, eights...)
   - Advantages: splitting is trivial (no carry complications), addition of sub-results is well-understood in binary
   - Disadvantage: sequences are longer (32-bit number = 32 tokens vs. 10 tokens in decimal)

2. **Decimal with digit-level tokenisation**
   - Each token = one digit (0-9)
   - More compact but Karatsuba splitting is messier (carry handling in additions)
   - More directly comparable to existing work (Cho et al., Jelassi et al.)

3. **Base-256 or base-1024 (chunked binary)**
   - Each token encodes 8 or 10 bits
   - Good balance of compactness and clean splitting
   - Worth trying after binary works

**Recommendation:** Start with binary. It makes the recursive structure maximally clean. If the approach works in binary, adapt to decimal for comparability with prior work.

### Decision 2: Scratchpad Format

This is the most important design decision. You need to represent the Karatsuba recursion tree as a sequence of tokens.

**Important new context:** Sato et al. (ICML 2025) showed that the *order* of reasoning steps critically affects learnability. Test both depth-first and breadth-first, and consider reverse orderings.

**Option A: Depth-first trace (RECOMMENDED)**

For 8-bit × 8-bit multiplication of X × Y, the scratchpad would look like:

```
[INPUT] X_bits Y_bits
[SPLIT] X_hi X_lo Y_hi Y_lo          # Split into 4-bit halves
[SUB_MUL_0] X_lo × Y_lo              # Recurse: z0
  [SPLIT] ...                         # Split 4-bit numbers into 2-bit
  [SUB_MUL_0] ...                     # Base case: 2-bit × 2-bit → direct
  [SUB_MUL_2] ...
  [SUB_MUL_1] ...
  [COMBINE] → z0_result
[SUB_MUL_2] X_hi × Y_hi              # Recurse: z2
  ...
  [COMBINE] → z2_result
[ADD] (X_lo + X_hi), (Y_lo + Y_hi)   # Compute sums for z1
[SUB_MUL_1] sum_X × sum_Y            # Recurse: z1 (before subtraction)
  ...
  [COMBINE] → product
[SUB] product − z0 − z2 → z1_result  # Karatsuba subtraction trick
[COMBINE] z2 · B^n + z1 · B^(n/2) + z0  → final_answer
[OUTPUT] result_bits
```

Each [SPLIT], [ADD], [SUB], [COMBINE] step involves only bounded-size operations. Each [SUB_MUL] step recurses on numbers half the size.

**Advantages:** Natural representation of the algorithm. Each step has bounded token dependency. The model only needs to learn: split, small multiply (base case), add, subtract, shift-and-combine.

**Disadvantages:** Long sequences (the full trace of Karatsuba on 32-bit numbers is substantial). Training data generation is more complex.

**Option B: Breadth-first / level-by-level**

Process all sub-problems at the same recursion level before going deeper:

```
Level 0: X × Y → need z0, z1, z2
Level 1: compute X_lo × Y_lo, X_hi × Y_hi, (X_lo+X_hi) × (Y_lo+Y_hi)
Level 2: each of those splits into 3 more sub-problems (9 total)
...
Base level: all sub-problems are small enough to multiply directly
Combine level k-1: combine results from level k
...
Combine level 0: produce final answer
```

**Advantages:** All sub-problems at the same level have the same structure, which is more natural for a looped architecture (one loop iteration = one level). **SpiralFormer (Feb 2026) provides evidence that multi-resolution recursion schedules work well with looped architectures.**

**Disadvantages:** Requires holding many intermediate results simultaneously.

**Option C: Hybrid — loop over recursion levels (ALSO WORTH TRYING)**

Use the looped transformer's iterations to correspond to recursion levels. At each iteration:
- The model processes all sub-problems at the current level
- The residual stream carries the intermediate results forward
- Position encodings indicate which sub-problem each token belongs to

This is the most architecturally elegant but hardest to get right.

**Recommendation:** Start with Option A (depth-first trace, autoregressive generation). It's the most straightforward to implement and debug. If it works, try Option C for a cleaner architecture. **However, given SpiralFormer's success with multi-resolution recursion and LoopFormer's shortcut-consistency training, Option B/C may actually be more natural for looped architectures — try them early.**

### Decision 3: Position Encoding Scheme

Building on position coupling (Cho et al.), design **hierarchical position encodings** that capture the recursive structure:

**Proposed encoding: (bit_significance, recursion_depth, sub_problem_id, step_type)**

Each token gets a 4-tuple:
- **bit_significance**: which bit position this token represents (0 = LSB, n-1 = MSB). For Karatsuba, this tells the model which half a bit belongs to after splitting.
- **recursion_depth**: how deep in the recursion tree we are (0 = top level, log₂(n) = base case).
- **sub_problem_id**: which of the 3 sub-problems (z0, z1, z2) at this level.
- **step_type**: what operation is happening (SPLIT=0, MULTIPLY=1, ADD=2, SUB=3, COMBINE=4).

**Position coupling rule:** Tokens that correspond to the same bit significance in the same sub-problem at the same recursion level share the same position ID. This directly encodes the Karatsuba structure into attention.

**Alternative (simpler):** Just use bit_significance + a flag for recursion_depth. Explore whether the model can learn the sub-problem routing from the data alone.

**New alternative to consider:** Orthogonal function-based encodings (wavelets, Legendre polynomials) from the June 2025 theoretical analysis paper, which outperformed sinusoidal on synthetic sequence-to-sequence tasks.

### Decision 4: Looped Transformer Configuration

**Architecture:**
- **Layers:** 1-2 shared layers (following Fan et al., fewer layers generalise better for algorithmic tasks)
- **Heads:** 4-8 attention heads
- **Hidden dim:** 128-256
- **Loop count:** Adaptive, with a learned halting mechanism or fixed at log₂(n) + constant
- **Timestep encoding:** Following Saunshi et al. (ICLR 2025), add a timestep embedding that tells the model which iteration it's on (critical for knowing the current recursion depth). Their work proves this enables "latent thought" simulation equivalent to explicit CoT.

**New architectural options from 2025-2026 looped transformer work:**

- **LoopFormer's shortcut-consistency (Feb 2026):** Train with varying loop depths and a shortcut modulation mechanism. This would let the model learn to handle different recursion depths robustly. The "time t and step size dt" conditioning is directly applicable to our recursion depth problem.
- **SpiralFormer's multi-resolution recursion (Feb 2026):** Early iterations process at coarse resolution (global structure), later iterations refine at token resolution. Maps naturally to our recursion: early loops handle top-level splits, later loops handle base cases.
- **PLT's cross-loop parallelism (Oct 2025):** At inference time on longer sequences, can parallelise across loop iterations for faster evaluation. Worth implementing for the 64-bit and 128-bit test cases.

**Key design choice — how loops map to recursion:**

**Option 1: One loop = one recursion level (top-down then bottom-up)**
- First log₂(n) iterations: decompose (split, compute sums)
- Next log₂(n) iterations: recombine (base-case multiply, then combine upward)
- Total iterations: ~2 · log₂(n)

**Option 2: One loop = one complete step of the trace**
- More iterations but each is simpler
- Easier to train (each step is a very local computation)

**Recommendation:** Start with Option 2 (simpler steps, more iterations). Transition to Option 1 if training is stable enough. **Consider LoopFormer's shortcut-consistency training from the start — it helps bridge the gap between different loop depths.**

### Decision 5: Base Case

At the bottom of the recursion, you need to multiply small numbers directly. Options:

- **1-bit × 1-bit** (literally AND gate): Simplest, but deepest recursion
- **4-bit × 4-bit**: Can be done in one transformer forward pass (only 256 possible products). Reduces recursion depth by 2.
- **8-bit × 8-bit**: Reduces recursion further but requires the model to learn harder base cases.

**Recommendation:** Start with 4-bit base case. This means:
- Training on 8-bit × 8-bit = 1 level of Karatsuba recursion (split 8→4, three 4-bit multiplies, combine)
- Testing on 16-bit × 16-bit = 2 levels (split 16→8→4)
- Testing on 32-bit × 32-bit = 3 levels
- Testing on 64-bit × 64-bit = 4 levels

If 4-bit base case works, push to 2-bit or 1-bit for cleaner theoretical properties.

---

## Development Workflow & Compute Setup

### Colab-Based Development (No Local GPU)

Since you don't have a physical GPU, all training runs on Google Colab. This shapes the entire workflow.

**Recommended Colab setup:**

| Phase | Hardware | Cost |
|-------|----------|------|
| Data generation & debugging | CPU or free T4 | Free |
| Phase 1-2 (prototyping, 4-bit/8-bit training) | Free T4 (16GB) | Free |
| Phase 3 (Karatsuba training, hyperparameter sweeps) | Colab Pro L4 (24GB) or A100 | ~$10/mo Pro |
| Phase 4-5 (ablations, multiple runs) | Colab Pro A100 (40GB) | ~$10/mo Pro |
| Phase 6 (interpretability, attention analysis) | Any GPU | Free/Pro |

**Why this works:** A 1-5M param model in fp32 uses ~20MB. Even a free T4 (16GB VRAM) gives you headroom for batch sizes of 256-1024+. The bottleneck is session time limits (12h free, longer on Pro), not memory.

**Colab GPU availability (as of 2026):**
- Free: T4 (16GB, Turing) — not guaranteed, 15-30 hrs/week
- Pro ($10/mo): T4, L4 (24GB, Ada Lovelace), A100 (40/80GB) — better availability
- Pro+ ($50/mo): Priority A100 access — use if ablation runs are bottlenecked
- TPU v2/v3 (8 cores) — free tier, excellent with JAX (see below)

### Human-in-the-Loop Workflow

Since Claude can write code but you need to execute on Colab, the workflow is:

1. **Claude writes code locally** → you copy to Colab notebook or push to GitHub
2. **You run on Colab** → copy results/logs back
3. **Claude analyses results** → suggests next experiment

**To make this efficient:**

- **Use a GitHub repo as the bridge.** Claude writes `.py` files locally. You `git clone` in Colab and `!pip install -e .` to get updates. Push results/logs to a `results/` branch or upload to Google Drive.
- **Structure code as importable modules, not notebook cells.** This avoids copy-paste hell. Notebooks should be thin wrappers:
  ```python
  # Colab notebook cell:
  !git clone https://github.com/you/karatsuba-transformers
  !pip install -e karatsuba-transformers
  from karatsuba_transformers import train, evaluate, generate_data
  train.run(config="configs/8bit_karatsuba.yaml")
  ```
- **Checkpoint aggressively to Google Drive.** Colab sessions die. Save every N steps:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  CHECKPOINT_DIR = '/content/drive/MyDrive/karatsuba_checkpoints/'
  ```
- **Use Weights & Biases for experiment tracking.** `wandb.init()` works in Colab and persists across sessions. Track loss curves, accuracy, loop counts, etc.

### Code Structure

```
karatsuba-transformers/
├── configs/                    # YAML experiment configs
│   ├── 4bit_base_case.yaml
│   ├── 8bit_karatsuba.yaml
│   ├── 8bit_school_baseline.yaml
│   └── sweep_positions.yaml
├── src/
│   ├── data/
│   │   ├── karatsuba_trace.py  # Karatsuba trace generator
│   │   ├── school_trace.py     # School algorithm trace generator
│   │   ├── tokenizer.py        # Token/position encoding
│   │   └── dataset.py          # Dataset class
│   ├── model/
│   │   ├── looped_transformer.py  # Core model (JAX/Equinox or PyTorch)
│   │   ├── position_encoding.py   # Hierarchical position encodings
│   │   └── halting.py             # Adaptive computation time
│   ├── training/
│   │   ├── train.py            # Training loop
│   │   ├── evaluate.py         # Eval + length generalisation tests
│   │   └── curriculum.py       # Curriculum learning logic
│   └── analysis/
│       ├── attention_viz.py    # Attention pattern visualisation
│       ├── mechanistic.py      # Mechanistic interpretability tools
│       └── metrics.py          # Per-digit, per-level accuracy
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_train_karatsuba.ipynb
│   └── 04_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## Performance & Speed Optimisation

### Framework Choice: JAX + Equinox (RECOMMENDED) or PyTorch + torch.compile

**Primary recommendation: JAX + Equinox + Optax**

For a small custom transformer trained on Colab, JAX is the best choice because:

1. **Colab TPU support is free and powerful.** Colab provides TPU v2/v3 with 8 cores for free. JAX is the native framework for TPUs. A 1-5M param model on 8 TPU cores with data parallelism will train very fast.
2. **XLA compilation via `jax.jit`** performs full graph analysis, fuses operations, eliminates redundant computation. For a small fixed-architecture model that you run millions of steps, the compilation overhead is amortised away.
3. **`jax.vmap`** for automatic batching — write your model for a single example, then vmap over batches. Cleaner code and XLA optimises the batched version.
4. **`jax.pmap`** for multi-device data parallelism across TPU cores — essentially free 8× throughput.
5. **Equinox** (by Patrick Kidger) is the best JAX neural network library for custom models: PyTorch-like syntax, models are just PyTrees, no framework magic, composes cleanly with jit/grad/vmap.

**Secondary option: PyTorch + `torch.compile`**

If you're more comfortable with PyTorch:
- Use `torch.compile()` to JIT-compile the training loop. As of 2025-2026, it provides significant speedups for standard transformer architectures.
- Use `torch.nn.functional.scaled_dot_product_attention()` which auto-dispatches to FlashAttention/memory-efficient attention.
- Use `torch.cuda.amp` for mixed precision on GPU (fp16 on T4, bf16 on A100/L4).
- nanoGPT is a good starting point but will need significant modifications for looped architecture.

**Speed comparison for small models:**

| Setup | Approx. Relative Speed | Notes |
|-------|----------------------|-------|
| JAX + TPU v2/v3 (8 cores, free Colab) | **1.0× (baseline, fastest)** | Best free option |
| JAX + Colab T4 GPU | 0.5-0.7× | Good, XLA helps |
| PyTorch + torch.compile + T4 | 0.4-0.6× | Decent, some compile overhead |
| PyTorch eager + T4 | 0.2-0.3× | Slowest, avoid |
| JAX + Colab A100 (Pro) | 2-4× vs TPU baseline | Overkill but fast |

### Specific Optimisation Techniques

**1. Mixed precision training:**
- On **TPU**: Use bf16 natively — TPUs are optimised for it. In JAX: `jax.default_matmul_precision('bfloat16')` or use `jnp.bfloat16` dtypes.
- On **T4 GPU**: Use fp16 with loss scaling (T4 doesn't support bf16). In JAX: use `jmp` library for mixed precision. In PyTorch: `torch.cuda.amp.autocast()`.
- On **A100/L4 GPU**: Use bf16, well-supported on Ampere+.
- **Caveat:** For very small models (<1M params), mixed precision may not help throughput much (small matrices don't saturate Tensor Cores). Profile and compare. The memory savings are irrelevant since the model already fits easily.

**2. Data pipeline — don't let it be the bottleneck:**
- Pre-generate all training data and store as memory-mapped arrays (numpy `.npy` or TFRecord for TPU).
- For 65K examples of 8-bit Karatsuba traces, the entire dataset fits in RAM. Load it all upfront.
- Use JAX's `jax.random` for on-the-fly data augmentation if needed (e.g., random bit-width padding).
- On TPU: use `tf.data` pipeline → JAX for efficient data loading.

**3. Compilation and JIT tips:**
- In JAX, `jax.jit` the entire training step (forward + loss + backward + optimizer update) as one compiled function. This avoids Python overhead between steps.
- Use `donate_argnums` to reuse memory for state updates.
- Use `jax.lax.scan` for the looped transformer iterations instead of a Python for-loop — this compiles the loop into XLA, avoiding re-tracing.
- For adaptive loop count: use `jax.lax.while_loop` with a halting condition, or `jax.lax.scan` with a fixed max and masking.

**4. FlashAttention:**
- On GPU with PyTorch: FlashAttention 2.x works on A100/L4 (Ampere+). On T4 (Turing), use FlashAttention 1.x or PyTorch's built-in SDPA.
- On TPU with JAX: XLA already fuses attention operations efficiently; explicit FlashAttention isn't needed. But consider using `jax.nn.dot_product_attention` which is optimised.
- For our small model with short-to-medium sequences (8-bit traces ~ 100-200 tokens, 32-bit traces ~ 1000-2000 tokens), FlashAttention helps mostly at the longer test lengths.

**5. Gradient accumulation for effective batch size:**
- If batch size is limited by sequence length at test-time evaluation, use gradient accumulation during training to simulate larger batches.
- In JAX: accumulate gradients with `jax.tree.map(jnp.add, grads_accum, grads)` over microbatches.

**6. Custom CUDA kernels (NOT recommended for this project):**
- Ninja-compiled CUDA kernels (via `torch.utils.cpp_extension`) can provide 1.5-2× speedups for specific operations (e.g., custom RMSNorm, fused attention).
- **However:** For a 1-5M param model, the engineering overhead isn't worth it. Use framework-provided fused operations instead. Custom kernels matter at scale (100M+ params), not here.
- **Exception:** If you find a specific bottleneck in the looped architecture (e.g., the loop overhead itself), a custom kernel might help. Profile first.

### JAX Implementation Skeleton

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

class LoopedTransformerBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    ffn: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    timestep_embed: eqx.nn.Embedding  # iteration number → embedding

    def __call__(self, x, timestep, mask=None):
        t_emb = self.timestep_embed(timestep)
        x = x + t_emb  # condition on loop iteration
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x

class KaratsubaTransformer(eqx.Module):
    embed: eqx.nn.Embedding
    pos_encode: HierarchicalPositionEncoding  # custom
    block: LoopedTransformerBlock  # single block, looped
    output_head: eqx.nn.Linear
    max_loops: int

    def __call__(self, tokens, positions, n_loops):
        x = self.embed(tokens) + self.pos_encode(positions)

        # Use jax.lax.scan for compiled loop
        def loop_body(carry, timestep):
            x = carry
            x = self.block(x, timestep)
            return x, None

        x, _ = jax.lax.scan(loop_body, x, jnp.arange(n_loops))
        return self.output_head(x)

# JIT-compile the entire training step
@eqx.filter_jit
def train_step(model, opt_state, batch):
    def loss_fn(model):
        logits = jax.vmap(model)(batch['tokens'], batch['positions'], batch['n_loops'])
        return cross_entropy(logits, batch['targets'])

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# For TPU: use pmap for data parallelism across 8 cores
@jax.pmap
def train_step_pmap(model, opt_state, batch):
    return train_step(model, opt_state, batch)
```

### Libraries & Versions

```
# requirements.txt
jax[tpu]>=0.5.2        # or jax[cuda12] for GPU
equinox>=0.11.0
optax>=0.2.0
jaxtyping>=0.2.0       # type annotations for JAX arrays
wandb>=0.19.0
numpy>=1.26.0
pyyaml>=6.0
matplotlib>=3.8.0      # for attention visualisation
einops>=0.8.0          # readable tensor operations

# Alternative PyTorch stack:
# torch>=2.5.0
# flash-attn>=2.8.0    # only on Ampere+ GPUs
```

---

## Experimental Plan

### Phase 0: Environment Setup & Data Generation

**Goal:** Get the full pipeline working end-to-end on Colab before training anything meaningful.

1. **Set up repo structure** (Claude writes locally, you clone on Colab)
2. **Implement Karatsuba trace generator** — write and test locally (pure Python, no GPU needed)
3. **Implement tokenizer and position encoding** — test on a few examples, verify traces are correct
4. **Build a tiny model** (d=32, 1 layer, 2 heads) and verify it trains on 4-bit × 4-bit on CPU
5. **Deploy to Colab** — verify JAX/Equinox setup, GPU/TPU detection, checkpoint saving to Drive

**Validation criteria:** Can generate a Karatsuba trace for 8-bit × 8-bit, tokenize it, feed it through the model, compute loss, backprop. No accuracy needed yet.

### Phase 1: Base Case Training

Train the model on **4-bit × 4-bit multiplication only** (no recursion, just the base case).

- All 256 pairs (exhaustive)
- Target: 100% exact-match accuracy (the model must learn to multiply small numbers perfectly)
- This is the foundation — if the model can't do 4-bit × 4-bit, nothing else works
- Quick iteration: should train in minutes on T4

### Phase 2: Baseline — School Algorithm Scratchpad

Before testing the Karatsuba approach, establish a baseline:

1. Same looped transformer architecture
2. Same binary representation
3. But scratchpad encodes the school algorithm (partial products + shifted addition)
4. Standard position coupling (bit significance only)

**Purpose:** Isolate the effect of the algorithm decomposition. If the Karatsuba scratchpad beats the school scratchpad on the same architecture, you've shown that algorithm choice matters for length generalisation.

### Phase 3: Karatsuba Scratchpad Training

Train the looped transformer on 8-bit × 8-bit Karatsuba traces.

**Training details:**
- Autoregressive next-token prediction on the scratchpad trace
- Cross-entropy loss on each token
- AdamW optimiser (or Lion — faster convergence for transformers), cosine learning rate schedule
- Weight decay (important for grokking/generalisation, per Nanda et al.)
- Train until convergence on training set
- **Intermediate supervision:** Add auxiliary loss on each recursion level's output, not just the final answer. This is analogous to deep supervision in U-Nets and should reduce error accumulation.

**Curriculum strategy:**
- Start with 4-bit × 4-bit (base case only, no recursion)
- Add 8-bit × 8-bit (one recursion level)
- This teaches the model the base case first, then the recursive structure
- **New idea from Cai et al. (NeurIPS 2025):** Consider co-training with a longer addition task as an auxiliary objective — length generalisation can transfer across related tasks.

### Phase 4: Length Generalisation Evaluation

Test on unseen lengths. Report:
- **Exact-match accuracy** at each test length (16, 32, 64, 128 bits)
- **Per-digit accuracy** (to see where errors occur — early digits? late digits? carry propagation?)
- **Per-recursion-level accuracy** (does the model handle the first recursion level correctly but fail on the second?)

**Compare against:**
- School algorithm baseline (Phase 2)
- **Hou et al. (ICML 2025) Turing Programs results** — the new SOTA. They achieve 97%+ on 100-digit multiplication. Compare in terms of both accuracy and computational efficiency (number of steps/tokens required).
- Cho et al. (2025) results (if reproducible in binary)
- Standard transformer (non-looped) with same scratchpad

### Phase 5: Ablations

Systematically test each design decision:

1. **Position encoding ablation:** Full hierarchical encoding vs. bit-significance only vs. no position coupling
2. **Loop count ablation:** Fixed loops vs. adaptive halting. Does the model learn to use more loops for larger numbers?
3. **Base case size:** 1-bit vs. 4-bit vs. 8-bit base cases
4. **Scratchpad format:** Depth-first vs. breadth-first trace (per Sato et al., ordering matters)
5. **Architecture size:** Does a larger model help or hurt? (Cho et al. found shallower models generalise better)
6. **Binary vs. decimal:** Does the clean binary splitting advantage survive translation to decimal?
7. **LoopFormer shortcut-consistency:** Does shortcut modulation improve generalisation to deeper recursion?
8. **Auxiliary task co-training:** Does adding addition traces improve multiplication generalisation (per Cai et al.)?

### Phase 6: Mechanistic Interpretability (Optional but High-Value)

Following Nanda et al.'s approach:

1. **Inspect attention patterns:** Do attention heads at different loop iterations specialise for different parts of the Karatsuba algorithm? (One head for splitting, one for the addition trick, etc.)
2. **Fourier analysis of embeddings:** Do the number embeddings learn Fourier-like structure even without explicit Fourier features?
3. **Ablation in representation space:** What happens if you zero out specific directions in the residual stream at specific loop iterations?
4. **Compare learned algorithm to Karatsuba:** Does the model actually implement Karatsuba, or does it find a different recursive decomposition?
5. **Loop utilisation analysis:** Verify that the model uses different loop iterations for different recursion levels. Plot attention patterns per iteration to see if there's a clear phase transition between "decompose" and "recombine" phases.

This phase is what would make the paper publishable at a top venue. Understanding *what the model learns* is more interesting than just showing it generalises.

---

## Compute Requirements

**Minimal setup (Colab free tier):**
- Model: ~1-5M parameters (1-2 layer transformer, d=256, 4-8 heads)
- Training data: 65K examples of 8-bit × 8-bit Karatsuba traces
- Hardware: Free Colab T4 (16GB) or TPU v2/v3 (8 cores)
- Inference on test lengths: fast (small model)

**This is deliberately small.** The whole point is that a tiny model with the right structure should generalise. If you need a big model, the approach isn't working.

**Recommended setup for full experiments:**
- Colab Pro ($10/mo) for reliable A100 access during ablation sweeps
- Google Drive for checkpoints and results
- Weights & Biases (free tier) for experiment tracking
- GitHub for code versioning and Colab ↔ local sync

**Tools:**
- JAX + Equinox + Optax (primary) or PyTorch + torch.compile (fallback)
- Weights & Biases for experiment tracking
- Matplotlib / Plotly for attention visualisation

---

## What Success Looks Like

**Minimum viable result (workshop paper):**
- Show that Karatsuba scratchpad + looped transformer achieves >90% exact-match accuracy on 16-bit multiplication when trained on 8-bit (2× generalisation), outperforming school-algorithm baseline

**Strong result (conference paper):**
- Show ≥4× length generalisation (8-bit → 32-bit) with high accuracy
- Demonstrate that generalisation scales predictably with loop count (more loops = handles more recursion levels)
- Include ablations showing each component contributes
- Include mechanistic analysis showing the model learns something like Karatsuba
- **Compare favourably or complementarily to Hou et al. (ICML 2025)** — show that recursive decomposition achieves comparable accuracy with fewer computational steps

**Exceptional result (top venue):**
- Show ≥16× generalisation (8-bit → 128-bit)
- Generalise the approach to other divide-and-conquer algorithms (Strassen's matrix multiply, merge sort, FFT) using the same looped architecture
- Provide theoretical analysis of why recursive decomposition helps length generalisation (link to circuit complexity: bounded-depth recursion → log-depth circuits → TC⁰)
- Connect to Izzo et al.'s quantitative bounds — show that recursive decomposition satisfies their "simulation" condition for length generalisation

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Looped transformer training instability | High | High | Start with fixed loop count, add ACT later. Use progressive training (short loops → long loops). Use LoopFormer's shortcut-consistency training (Feb 2026). |
| Karatsuba scratchpad too long for context | Medium | Medium | Use chunked binary (base-256) to compress sequences. Or use 4-bit base case to limit recursion depth. Profile actual sequence lengths for 16/32/64-bit inputs. |
| Model memorises traces instead of learning algorithm | Medium | High | Weight decay (essential per Nanda et al.). Test on unseen number pairs at training length before testing generalisation. Monitor training vs. test loss gap. |
| Error accumulation across recursion levels | High | Medium | Add intermediate supervision (loss on each recursion level's output, not just final answer). This is analogous to deep supervision in U-Nets. |
| Colab session timeouts during long training runs | Medium | Medium | Checkpoint every 500 steps to Google Drive. Use `try/except` to save on interrupt. Break long runs into resumable segments. Consider Colab Pro for longer sessions. |
| Hou et al. (ICML 2025) already solves the problem | Medium | Medium | Our approach is complementary, not competing: they use O(n²) TM simulation, we use O(n^1.585) recursive decomposition. Even matching their accuracy with fewer steps is a contribution. |
| Approach works but doesn't beat Cho et al. | Medium | Medium | The contribution is still valid if you show algorithm choice matters, even if absolute numbers are comparable. The mechanistic analysis adds value regardless. |
| JAX learning curve slows progress | Medium | Low | Fallback to PyTorch + torch.compile. The model is simple enough that framework choice isn't critical. Equinox has PyTorch-like syntax to ease transition. |
| No one has done this because it doesn't work | Low-Medium | High | The theoretical argument is sound (bounded per-step computation + log-depth recursion). Saunshi et al. (ICLR 2025) prove looped transformers can simulate CoT, and Izzo et al. (2025) prove length generalisation follows from "simulability." If it fails, understanding *why* is itself a contribution. |

---

## Reading Order (Prioritised)

1. **Hou, Brandfonbrener et al. (ICML 2025)** — Turing Programs for multiplication (new SOTA, primary comparison) [arxiv.org/abs/2407.03310](https://arxiv.org/abs/2407.03310)
2. **Cho et al. (NeurIPS 2024)** — Position Coupling (foundation technique) [arxiv.org/abs/2405.20671](https://arxiv.org/abs/2405.20671)
3. **Fan et al. (ICLR 2025)** — Looped Transformers for Length Generalisation (architecture backbone)
4. **Saunshi et al. (ICLR 2025)** — Latent Thoughts / Looped Transformers (theoretical validation) [arxiv.org/abs/2502.17416](https://arxiv.org/abs/2502.17416)
5. **LoopFormer — Jeddi et al. (Feb 2026)** — Shortcut-consistency training (practical training technique) [arxiv.org/abs/2602.11451](https://arxiv.org/abs/2602.11451)
6. **SpiralFormer — Yu et al. (Feb 2026)** — Multi-resolution recursion (architecture idea) [arxiv.org/abs/2602.11698](https://arxiv.org/abs/2602.11698)
7. **Nanda et al. (ICLR 2023)** — Grokking / mechanistic interpretability (understanding what models learn)
8. **Cho et al. (ICLR 2025)** — Arithmetic Transformers (current position-coupling SOTA on multiplication) [arxiv.org/abs/2410.15787](https://arxiv.org/abs/2410.15787)
9. **Izzo et al. (Oct 2025)** — Quantitative Length Generalisation Bounds (theoretical backing) [arxiv.org/abs/2510.27015](https://arxiv.org/abs/2510.27015)
10. **Sato et al. (ICML 2025)** — CoT ordering for arithmetic (scratchpad design) [arxiv.org/abs/2506.23875](https://arxiv.org/abs/2506.23875)
11. **Cai et al. (NeurIPS 2025)** — Length Generalisation Transfer (cross-task transfer technique) [arxiv.org/abs/2506.09251](https://arxiv.org/abs/2506.09251)
12. **Karatsuba algorithm** — Review the algorithm itself until you can implement it from memory
13. **Jelassi et al. (2023)** — Length Generalisation in Arithmetic Transformers (baseline / context)
14. **Giannou et al. (ICML 2023)** — Looped Transformers as Programmable Computers (theory)
15. **Back de Luca et al. (NeurIPS 2025)** — NTK framework for exact arithmetic (theoretical backing) [arxiv.org/abs/2502.16763](https://arxiv.org/abs/2502.16763)
16. **JAX Training Cookbook** — Practical JAX training guide [docs.jax.dev/en/latest/the-training-cookbook.html](https://docs.jax.dev/en/latest/the-training-cookbook.html)
17. **Equinox docs** — PyTree-based neural networks in JAX [docs.kidger.site/equinox/](https://docs.kidger.site/equinox/)
