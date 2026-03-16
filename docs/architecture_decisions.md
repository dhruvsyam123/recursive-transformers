# Architecture Decisions: Rationale and References

This document explains the key architectural decisions for the Karatsuba-style recursive multiplication project, with justifications and references to the relevant literature.

---

## Decision 1: Binary Representation (First)

### Choice
Start with binary representation where each token is one bit (0 or 1).

### Rationale

1. **Clean splitting**: In Karatsuba, the first operation is splitting a number into its high and low halves. In binary, this is trivial: take the top n/2 bits and the bottom n/2 bits. No carry handling or rounding is needed during the split. In decimal, splitting a number in half is messier (you need to know the number of digits, and Karatsuba traditionally operates on equal-length halves).

2. **Shift operations are free**: The Karatsuba combine step requires multiplying by B^(n/2) (shifting left by n/2 positions). In binary, this is simply appending n/2 zeros. In the transformer's position encoding, this corresponds to an offset in the bit_significance dimension. No actual computation is needed.

3. **Addition and subtraction are simpler in binary**: Each digit is 0 or 1, carry propagation has only 2 states. The transformer only needs to learn binary addition, which is well-studied (Cho et al., Fan et al. have both shown transformers can learn binary/decimal addition with length generalization).

4. **Position encoding is natural**: Bit significance (LSB=0, MSB=n-1) maps directly to a linear position encoding. For Karatsuba, the bit position immediately tells you which half the bit belongs to after a split (position < n/2 -> low half, position >= n/2 -> high half).

5. **Base case is tiny**: A 4-bit * 4-bit multiplication has only 256 possible input pairs, all producing at most 8-bit results. The base case is trivially memorizable.

### Trade-offs

- **Longer sequences**: A 32-bit number requires 32 tokens in binary vs. ~10 tokens in decimal. This increases sequence length by ~3x.
- **Less comparable to prior work**: Most prior work (Cho et al., Hou et al., Jelassi et al.) uses decimal. Direct comparison requires either reimplementing their methods in binary or adapting our method to decimal.
- **Plan**: Start binary, adapt to decimal after validating the approach. Binary is the "proof of concept" representation; decimal is for "paper-ready" comparisons.

### References
- Karatsuba's original algorithm naturally operates on any base; binary is the cleanest
- McLeish et al. (NeurIPS 2024): per-digit embeddings work for both binary and decimal
- Cho et al. (NeurIPS 2024): position coupling assigns positions by digit significance, which is equally natural in binary

---

## Decision 2: Depth-First Trace Ordering (Primary)

### Choice
Use depth-first traversal of the Karatsuba recursion tree for the scratchpad format. Complete each sub-problem fully before moving to the next.

### Rationale

1. **Bounded working memory**: In a depth-first trace, when computing a base-case multiplication, all the information needed is local: the two small operands. The model does not need to hold multiple unfinished sub-problems in its context simultaneously. This keeps the attention pattern simple.

2. **Natural for autoregressive generation**: A depth-first trace can be generated left-to-right by the transformer. Each step depends only on previously generated tokens. A breadth-first trace would require computing all sub-problems at a level before any of them can produce results, requiring more complex scheduling.

3. **Error containment**: If the model makes an error in one sub-problem, the error is contained to that branch of the recursion tree. In a breadth-first trace, errors at one level propagate to all sub-problems at the next level.

4. **Supported by Sato et al. (ICML 2025)**: Their work on CoT ordering shows that the order of reasoning steps critically affects learnability. They found that for multiplication, reverse-digit ordering works best. Our depth-first trace naturally processes low-order bits (z0) before high-order bits (z2), which is a form of "reverse" ordering (LSB first).

5. **Simpler implementation**: The data generation code simply follows the natural recursion, printing each step as it is computed. No complex scheduling or buffering is needed.

### Alternative: Breadth-First / Level-by-Level

Breadth-first ordering processes all sub-problems at the same recursion level before going deeper. This has advantages for looped architectures:
- One loop iteration = one recursion level (cleaner mapping)
- All sub-problems at the same level have identical structure (easier for the model)
- **SpiralFormer (Feb 2026)** provides evidence that multi-resolution processing works well with looped architectures

**Plan**: Start with depth-first (simpler, safer). Try breadth-first as an ablation. If using the "one loop = one recursion level" architecture (Option C from the research plan), breadth-first may be more natural.

### References
- Sato et al. (ICML 2025, arxiv 2506.23875): CoT ordering matters for arithmetic
- SpiralFormer (Feb 2026, arxiv 2602.11698): Multi-resolution recursion schedules
- Hou et al. (ICML 2025): Their Turing Program trace is inherently sequential (flat), not tree-structured

---

## Decision 3: Hierarchical Position Encoding

### Choice
Use a 4-tuple position encoding: (bit_significance, recursion_depth, sub_problem_id, step_type).

### Rationale

1. **Extends position coupling to recursive structure**: Cho et al. (NeurIPS 2024) showed that coupling positions by digit significance dramatically improves addition generalization. Our hierarchical encoding extends this idea to the recursive structure of Karatsuba by adding recursion_depth, sub_problem_id, and step_type.

2. **Enables the model to know "where it is" in the recursion**:
   - **bit_significance**: Which bit position (0=LSB). Tells the model which half a bit belongs to after splitting.
   - **recursion_depth**: How deep in the recursion tree (0=top, log(n)=base). Tells the model whether to split further or compute a base case.
   - **sub_problem_id**: Which of the 3 sub-problems (z0, z1, z2). Routes information between related sub-problems.
   - **step_type**: What operation (SPLIT=0, MUL=1, ADD=2, SUB=3, COMBINE=4). Tells the model which computation to perform.

3. **Position coupling rule**: Tokens with the same bit significance in the same sub-problem at the same recursion level share the same position ID. This directly encodes the Karatsuba structure into attention: the model automatically attends to structurally relevant tokens.

4. **Enables length generalization**: When the input length doubles, the recursion depth increases by 1 and new sub-problem IDs are added, but the position encoding scheme remains the same. The model can generalize because it has seen each type of step (SPLIT, MUL, ADD, SUB, COMBINE) at each relative depth during training.

5. **Theoretical backing**: Izzo et al. (Oct 2025) prove that length generalization occurs when behavior on longer sequences can be "simulated" by shorter ones. Our hierarchical position encoding ensures that each step at depth d of a longer problem looks identical to the same step at depth d of a shorter problem -- satisfying their simulability condition.

### Alternative: Simpler Encoding

A simpler alternative uses only bit_significance + recursion_depth, letting the model learn sub-problem routing from the data. This reduces the amount of human-designed structure in the encoding but may be harder to learn.

### Alternative: Orthogonal Function Encodings

A June 2025 paper on positional encoding theory (arxiv 2506.06398) proposes using orthogonal function families (wavelets, Legendre polynomials) instead of sinusoidal encodings. Wavelets are naturally hierarchical and might be a good fit for our recursive structure. Worth exploring as an ablation.

### References
- Cho et al. (NeurIPS 2024, arxiv 2405.20671): Position coupling foundation
- McLeish et al. (NeurIPS 2024, arxiv 2405.17399): Per-digit position embeddings
- Izzo et al. (Oct 2025, arxiv 2510.27015): Simulability condition for generalization
- Positional encoding theory (June 2025, arxiv 2506.06398): Orthogonal function families

---

## Decision 4: JAX + Equinox

### Choice
Use JAX with Equinox (and Optax for optimization) as the primary framework. PyTorch + torch.compile as fallback.

### Rationale

1. **Free Colab TPU support**: Google Colab provides free TPU v2/v3 access (8 cores). JAX is the native framework for TPUs. A 1-5M parameter model on 8 TPU cores with data parallelism trains very fast. This is the best free compute option.

2. **XLA compilation via jax.jit**: Full graph analysis, operation fusion, and elimination of redundant computation. For a small fixed-architecture model run for millions of steps, the one-time compilation overhead is negligible.

3. **jax.vmap for automatic batching**: Write the model for a single example, then vmap over batches. Cleaner code and XLA optimizes the batched version.

4. **jax.lax.scan for looped execution**: The looped transformer iterations can be compiled into a single XLA loop using jax.lax.scan, avoiding Python loop overhead. This is critical for efficiency: our model may need 10-20+ loop iterations, and each Python-level loop call adds overhead.

5. **jax.pmap for multi-device parallelism**: Essentially free 8x throughput across TPU cores with a single decorator.

6. **Equinox's design philosophy**: Models are plain PyTrees (Python objects), no framework magic. This makes custom architectures (like our hierarchical position encoding and looped transformer) easy to implement and debug. The PyTorch-like syntax eases the learning curve.

7. **jaxtyping for safety**: Type annotations for JAX arrays catch shape errors at definition time, which is essential for the complex position encoding scheme.

### Trade-offs

- **Learning curve**: JAX's functional programming paradigm is different from PyTorch's imperative style. Debugging can be harder (errors inside jit-compiled functions are less informative).
- **Smaller ecosystem**: Fewer pre-built components compared to PyTorch. We'll need to implement more from scratch.
- **PyTorch fallback**: If JAX proves too difficult, PyTorch + torch.compile provides 80% of the performance benefits with a more familiar API. The model is simple enough that framework choice isn't critical.

### References
- Fan et al. (ICLR 2025): Their looped transformer code is in PyTorch, but the architecture translates directly to JAX
- Equinox documentation: docs.kidger.site/equinox/
- JAX Training Cookbook: docs.jax.dev/en/latest/the-training-cookbook.html
- Research plan Section on Performance & Speed Optimization for detailed framework comparison

---

## Decision 5: 4-Bit Base Case

### Choice
Recurse until operands are 4 bits wide, then multiply directly (base case). The model must learn 4-bit * 4-bit = 8-bit multiplication as a "lookup" (256 possible input pairs).

### Rationale

1. **Manageable recursion depth**: With a 4-bit base case:
   - 8-bit training: 1 level of Karatsuba recursion (split 8->4, three 4-bit multiplies, combine)
   - 16-bit testing: 2 levels
   - 32-bit testing: 3 levels
   - 64-bit testing: 4 levels
   - 128-bit testing: 5 levels

   This gives a clear ladder for testing generalization: each doubling of input size adds exactly one recursion level.

2. **Small enough to memorize**: 4 * 4 = 16 input bits, 256 possible input pairs. The transformer can memorize all base cases perfectly. If we used 8-bit base case (65K pairs), memorization would be harder and might compete with learning the recursive structure.

3. **Large enough to be non-trivial**: A 1-bit base case (AND gate) would require the deepest recursion (log_2(n) levels for n-bit numbers). While theoretically cleanest, it means 8-bit training needs 3 recursion levels and 27 base-case multiplications. The scratchpad becomes very long. The 4-bit base case keeps the scratchpad manageable for 8-bit training while still testing generalization.

4. **Single recursion level for initial training**: With 4-bit base, training on 8-bit only requires learning ONE level of Karatsuba structure plus the base case. This is the simplest setting that demonstrates the recursive approach. Training complexity increases gradually as we test on longer inputs.

5. **Practical scratchpad length**: For 8-bit training with 4-bit base, the depth-first trace has approximately 50 tokens. For 16-bit testing with 4-bit base (2 levels), approximately 150 tokens. These are well within transformer context window limits.

### Trade-offs

- **The model must learn a non-trivial base case**: 4-bit * 4-bit is 256 cases producing 8-bit results. The model needs enough capacity to represent this lookup. A 128- or 256-dimensional embedding should be sufficient.
- **Reduces the "purity" of the recursion**: The theoretical argument for length generalization is strongest when every level of recursion is identical. With a 4-bit base case, the bottom level is different (lookup vs. recursive split). However, this is standard in practice -- even real Karatsuba implementations switch to schoolbook multiplication for small inputs.

### Future exploration

- Try 2-bit base case for cleaner theoretical properties (only 4 possible products: 0*0, 0*1, 1*0, 1*1... wait, 2-bit gives 0-3, so 16 products)
- Try 1-bit base case (AND gate) if deeper recursion is needed for testing
- Try 8-bit base case if training stability is an issue (fewer recursion levels during training)

### References
- Standard Karatsuba implementations use base cases of 32-64 bits in practice
- Nanda et al. (ICLR 2023): Even a lookup table (modular addition mod 113 = 113 cases) can be learned by a small transformer with interesting internal structure
- Research plan Phase 5 Ablation 3: base case size comparison is a planned experiment

---

## Decision 6: Looped Transformer with Timestep Encoding

### Choice
Use a 1-2 layer transformer block with shared weights, looped multiple times, with a learned timestep embedding added at each iteration.

### Rationale

1. **Weight sharing enables length generalization**: Fan et al. (ICLR 2025) showed that weight-shared (looped) transformers with adaptive depth generalize much better than fixed-depth models on iterative tasks. The same weights applied at each recursion level means the model learns a generic "recursion step" rather than level-specific transformations.

2. **Adaptive loop count for variable recursion depth**: At training time, use log_2(n_train / base_size) * 2 loops (down and up through the recursion tree). At test time, increase loops proportionally for longer inputs. This is the mechanism for length generalization.

3. **Timestep encoding is critical**: Saunshi et al. (ICLR 2025) show that looped transformers need to know which iteration they are on to generate "latent thoughts" effectively. Without timestep encoding, all iterations would behave identically, which prevents the model from doing different things at different recursion depths (decompose vs. recombine).

4. **Small model is intentional**: The research plan targets 1-5M parameters. Cho et al. found that shallower models generalize better for algorithmic tasks. A small model forces the model to learn a general algorithm rather than memorize input-output mappings.

5. **Two architectural options for how loops map to recursion**:
   - **Option 1: One loop = one recursion level** (top-down then bottom-up). Total: ~2*log_2(n) loops. Cleaner but requires the model to handle both decomposition and recombination phases.
   - **Option 2: One loop = one trace step**. More loops but each is simpler. Easier to train initially.

### Configuration Starting Point

```
layers: 2 (shared, looped)
heads: 4
hidden_dim: 256
max_loops: 20 (enough for 128-bit with 4-bit base: 5 levels * 2 phases * ~2 buffer)
timestep_embed_dim: 32
vocab: {0, 1, SPLIT, MUL_BASE, ADD, SUB, COMBINE, INPUT, OUTPUT, PAD}
```

### References
- Fan et al. (ICLR 2025, arxiv 2409.15647): Looped transformers for length generalization
- Saunshi et al. (ICLR 2025, arxiv 2502.17416): Theoretical validation, latent thoughts, timestep importance
- Dehghani et al. (2018): Universal Transformers, original weight-sharing + ACT
- LoopFormer (Feb 2026, arxiv 2602.11451): Shortcut-consistency training for robust loop-depth handling
- Giannou et al. (ICML 2023): Looped transformers as programmable computers (theoretical foundation)

---

## Summary: Decision Dependencies

```
Binary Representation
    |
    v
4-bit Base Case -----------> Manageable scratchpad lengths
    |                              |
    v                              v
Depth-first Trace ---------> Natural for autoregressive generation
    |                              |
    v                              v
Hierarchical Position Encoding --> Enables length generalization
    |
    v
Looped Transformer + Timestep --> Adaptive depth for variable recursion
    |
    v
JAX + Equinox --> Efficient implementation on free Colab TPUs
```

Each decision builds on the previous ones. Binary makes splitting clean, which makes the depth-first trace clean, which makes the hierarchical position encoding well-defined, which makes the looped transformer able to generalize to deeper recursion. JAX + Equinox is the implementation choice that makes this all efficient to train.
