# Experiment Log

## Template

Each experiment entry should follow this format:

```
### EXP-XXX: [Experiment Name]

**Date:** YYYY-MM-DD
**Phase:** [0-6]
**Config:**
- Model: [layers, heads, hidden_dim, loops]
- Data: [input bits, base case bits, num_examples]
- Training: [optimizer, lr, weight_decay, batch_size, epochs/steps]
- Hardware: [GPU/TPU type, Colab tier]

**Hypothesis:** [What do you expect to happen and why?]

**Result:** [PENDING | SUCCESS | PARTIAL | FAILURE]
- Training accuracy: X%
- Test accuracy at [length]: X%
- Training time: X hours
- Key metric: [whatever is most relevant]

**Notes:** [Observations, surprises, next steps]

**Artifacts:**
- Checkpoint: [path or Google Drive link]
- W&B run: [link]
- Plots: [paths]
```

---

## Phase 0: Environment Setup & Data Generation

### EXP-001: End-to-End Pipeline Verification

**Date:** PENDING
**Phase:** 0
**Config:**
- Model: 2 layers, 2 heads, hidden_dim=32 (tiny, for verification only)
- Data: 4-bit * 4-bit, all 256 pairs, no recursion (base case only)
- Training: AdamW, lr=1e-3, weight_decay=0.01, batch_size=64, 100 steps
- Hardware: CPU (local or free Colab)

**Hypothesis:** The pipeline (data generation -> tokenization -> model forward -> loss -> backward) should run without errors. Not expecting meaningful accuracy -- just testing that everything connects.

**Result:** PENDING

**Notes:** Validation criteria: Can generate a Karatsuba trace for 8-bit * 8-bit, tokenize it, feed it through the model, compute loss, and backprop. No accuracy needed.

---

### EXP-002: Karatsuba Trace Generation Verification

**Date:** PENDING
**Phase:** 0
**Config:**
- Pure Python, no model
- Generate depth-first Karatsuba traces for 8-bit * 8-bit with 4-bit base case
- Verify correctness on all 65,536 possible 8-bit * 8-bit pairs

**Hypothesis:** All traces should produce correct final results. Trace lengths should be consistent for same-size inputs.

**Result:** PENDING

**Notes:** Key things to verify:
- Carry overflow handling in (X_lo + X_hi) sums
- Correct bit ordering (LSB first vs MSB first)
- Consistent padding/alignment
- Trace token count statistics (min, max, mean, std)

---

### EXP-003: Colab Deployment Test

**Date:** PENDING
**Phase:** 0
**Config:**
- Same as EXP-001 but on Colab
- Test: JAX + Equinox installation, GPU/TPU detection, checkpoint save to Drive
- Hardware: Free Colab T4 GPU and/or TPU v2

**Hypothesis:** JAX should detect the accelerator, the model should train faster than CPU, and checkpoints should save to Drive.

**Result:** PENDING

**Notes:** Test both GPU and TPU paths. Record setup time (pip install, compilation).

---

## Phase 1: Base Case Training

### EXP-010: 4-bit * 4-bit Base Case (Exhaustive)

**Date:** PENDING
**Phase:** 1
**Config:**
- Model: 2 layers, 4 heads, hidden_dim=128, loops=1 (no looping needed)
- Data: All 256 pairs of 4-bit * 4-bit, split 80/20 train/test
- Training: AdamW, lr=3e-4, weight_decay=0.1, batch_size=128, train until 100% accuracy
- Hardware: Free Colab T4

**Hypothesis:** The model should achieve 100% exact-match accuracy on all 256 pairs. This is essentially a lookup table. If this fails, the model is too small or the tokenization is wrong.

**Result:** PENDING

**Notes:** This is the foundation. Track:
- Steps to convergence
- Whether grokking occurs (memorize training set quickly, then slowly generalize to held-out set)
- Attention patterns at convergence

---

### EXP-011: 4-bit * 4-bit with Varying Model Sizes

**Date:** PENDING
**Phase:** 1
**Config:**
- Models: (1 layer, 2 heads, d=64), (2 layers, 4 heads, d=128), (2 layers, 4 heads, d=256)
- Data: Same as EXP-010
- Training: Same as EXP-010

**Hypothesis:** All models should reach 100%. Smaller models may take longer (grokking). Larger models may memorize faster but also generalize faster due to weight decay pressure.

**Result:** PENDING

**Notes:** Use this to select the model size for Phase 2-3.

---

## Phase 2: Baseline -- School Algorithm Scratchpad

### EXP-020: School Algorithm Scratchpad (8-bit * 8-bit)

**Date:** PENDING
**Phase:** 2
**Config:**
- Model: [best from Phase 1], loops=[TBD based on school algorithm trace length]
- Data: 8-bit * 8-bit, school algorithm scratchpad, 10K training examples
- Training: AdamW, lr=3e-4, weight_decay=0.1, batch_size=128
- Hardware: Free Colab T4 or TPU

**Hypothesis:** The model should learn to generate correct school-algorithm scratchpads for 8-bit multiplication. This establishes the baseline for comparison with Karatsuba.

**Result:** PENDING

**Notes:** Track:
- Training accuracy vs. exact-match accuracy (the model may get individual tokens right but mess up the full answer)
- Per-step accuracy (which steps of the school algorithm are hardest?)

---

### EXP-021: School Algorithm Length Generalization (8-bit -> 16-bit)

**Date:** PENDING
**Phase:** 2
**Config:**
- Model: Trained in EXP-020
- Data: Test on 16-bit * 16-bit (with appropriately extended school algorithm scratchpad)
- Increase loop count proportionally at test time

**Hypothesis:** Length generalization will be limited (1-2x at best) because the school algorithm requires each output digit to attend to a growing number of input digits.

**Result:** PENDING

**Notes:** This is the baseline that Karatsuba should beat.

---

## Phase 3: Karatsuba Scratchpad Training

### EXP-030: Karatsuba Scratchpad (8-bit * 8-bit, depth-first)

**Date:** PENDING
**Phase:** 3
**Config:**
- Model: [best from Phase 1], loops=[2*1 + buffer = ~6-8 for 1 recursion level]
- Data: 8-bit * 8-bit, Karatsuba depth-first scratchpad with 4-bit base case, all 65K pairs or 10K sample
- Position encoding: Hierarchical (bit_significance, recursion_depth, sub_problem_id, step_type)
- Training: AdamW, lr=3e-4, weight_decay=0.1, batch_size=128, curriculum (4-bit base case first, then 8-bit)
- Hardware: Colab T4 or TPU

**Hypothesis:** The model should learn to generate correct Karatsuba traces. With hierarchical position encoding and the recursive structure, training should converge. May require curriculum learning (start with base cases only, then add the recursion level).

**Result:** PENDING

**Notes:** This is the core experiment. Track:
- Training curve shape (look for grokking pattern)
- Per-recursion-level accuracy (does the model get base cases right first?)
- Per-step-type accuracy (SPLIT, MUL_BASE, ADD, SUB, COMBINE)
- Scratchpad token accuracy vs. final answer accuracy

---

### EXP-031: Karatsuba with Intermediate Supervision

**Date:** PENDING
**Phase:** 3
**Config:**
- Same as EXP-030 but add auxiliary loss on each recursion level's output (not just the final answer)
- Loss = main_loss + alpha * sum(intermediate_losses)

**Hypothesis:** Intermediate supervision should speed up training and reduce error accumulation across recursion levels. This is analogous to deep supervision in U-Nets.

**Result:** PENDING

**Notes:** Compare training curves with and without intermediate supervision. Alpha values to try: 0.1, 0.3, 0.5, 1.0.

---

### EXP-032: Karatsuba with Curriculum Learning

**Date:** PENDING
**Phase:** 3
**Config:**
- Phase A: Train on 4-bit base cases only (1000 steps)
- Phase B: Add 8-bit Karatsuba traces with frozen base-case weights (or low LR on base-case parameters)
- Phase C: Full fine-tuning on 8-bit Karatsuba traces

**Hypothesis:** Curriculum should help by establishing a strong base case before asking the model to learn the recursion structure. Without curriculum, the model may struggle to learn base cases and recursion simultaneously.

**Result:** PENDING

**Notes:** Compare with EXP-030 (no curriculum).

---

## Phase 4: Length Generalization Evaluation

### EXP-040: Karatsuba Length Generalization (8-bit -> 16-bit)

**Date:** PENDING
**Phase:** 4
**Config:**
- Model: Best from Phase 3 (trained on 8-bit)
- Test data: 16-bit * 16-bit (2 recursion levels with 4-bit base case)
- Loop count: Increase from training value to accommodate 2 recursion levels
- Evaluate: 1000 random 16-bit pairs

**Hypothesis:** If the model has learned the recursive structure, it should generalize to 16-bit with >80% accuracy by using more loop iterations. This is the key test.

**Result:** PENDING

**Notes:** Track:
- Exact-match accuracy on full result
- Per-digit accuracy (which bits are wrong?)
- Per-recursion-level accuracy (does level 1 work but level 2 fail?)
- Comparison with school algorithm baseline (EXP-021)

---

### EXP-041: Karatsuba Length Generalization (8-bit -> 32-bit)

**Date:** PENDING
**Phase:** 4
**Config:**
- Model: Best from Phase 3
- Test data: 32-bit * 32-bit (3 recursion levels)
- Loop count: Increase for 3 levels

**Hypothesis:** This is a 4x generalization factor. If the model has truly learned Karatsuba, accuracy should be reasonable (>50%). Some error accumulation across 3 levels is expected.

**Result:** PENDING

---

### EXP-042: Karatsuba Length Generalization (8-bit -> 64-bit)

**Date:** PENDING
**Phase:** 4
**Config:**
- Model: Best from Phase 3
- Test data: 64-bit * 64-bit (4 recursion levels)
- Loop count: Increase for 4 levels

**Hypothesis:** 8x generalization. If this works at all (>10% accuracy), the approach is very promising.

**Result:** PENDING

---

### EXP-043: Karatsuba Length Generalization (8-bit -> 128-bit)

**Date:** PENDING
**Phase:** 4
**Config:**
- Model: Best from Phase 3
- Test data: 128-bit * 128-bit (5 recursion levels)
- Loop count: Increase for 5 levels

**Hypothesis:** 16x generalization. This would be an exceptional result. Expect degraded but non-trivial accuracy if the recursive structure is truly learned.

**Result:** PENDING

---

### EXP-044: Head-to-Head Comparison with Hou et al. (ICML 2025) Numbers

**Date:** PENDING
**Phase:** 4
**Config:**
- Convert our results to decimal equivalents for comparison
- 8-bit binary ~ 3-digit decimal; 16-bit ~ 5-digit; 32-bit ~ 10-digit; 64-bit ~ 19-digit; 128-bit ~ 39-digit
- Compare: accuracy and computational cost (total scratchpad tokens / loop iterations)

**Hypothesis:** Even if our raw accuracy is lower, we should use significantly fewer computational steps (O(n^1.585) vs O(n^2)).

**Result:** PENDING

---

## Phase 5: Ablations

### EXP-050: Position Encoding Ablation

**Date:** PENDING
**Phase:** 5
**Config:**
- A: Full hierarchical encoding (bit_significance, recursion_depth, sub_problem_id, step_type)
- B: bit_significance + recursion_depth only
- C: bit_significance only (like standard position coupling)
- D: Standard absolute positional encoding (no coupling)
- All trained on 8-bit, tested on 16-bit and 32-bit

**Hypothesis:** Full hierarchical > partial > bit-significance-only > standard. The recursion_depth and sub_problem_id components should be critical for length generalization.

**Result:** PENDING

---

### EXP-051: Loop Count Ablation

**Date:** PENDING
**Phase:** 5
**Config:**
- A: Fixed loops (always 8, even at test time)
- B: Adaptive loops (scale with input length)
- C: Adaptive loops with ACT (learned halting)
- All trained on 8-bit, tested on 16/32/64-bit

**Hypothesis:** Adaptive > ACT > Fixed. The model needs more loops for longer inputs. ACT should work but may not learn the optimal stopping point initially.

**Result:** PENDING

---

### EXP-052: Base Case Size Ablation

**Date:** PENDING
**Phase:** 5
**Config:**
- A: 1-bit base case (AND gate, deepest recursion)
- B: 2-bit base case (4 products)
- C: 4-bit base case (256 products) -- default
- D: 8-bit base case (65K products, shallowest recursion for 8-bit training = 0 levels)
- All with appropriate recursion depth adjustments

**Hypothesis:** 4-bit should be the sweet spot. 1-bit has too many recursion levels for 8-bit training. 8-bit has no recursion during training (just memorization), so it cannot generalize.

**Result:** PENDING

---

### EXP-053: Scratchpad Format Ablation

**Date:** PENDING
**Phase:** 5
**Config:**
- A: Depth-first trace (default)
- B: Breadth-first trace
- C: Reverse depth-first (process z2 before z0)
- All on same model/data

**Hypothesis:** Depth-first should work well based on Sato et al.'s findings. Breadth-first may be better for the looped architecture. Worth testing both.

**Result:** PENDING

---

### EXP-054: Architecture Size Ablation

**Date:** PENDING
**Phase:** 5
**Config:**
- A: 1 layer, 2 heads, d=64 (~100K params)
- B: 1 layer, 4 heads, d=128 (~500K params)
- C: 2 layers, 4 heads, d=128 (~1M params)
- D: 2 layers, 8 heads, d=256 (~5M params)
- E: 4 layers, 8 heads, d=256 (~10M params)

**Hypothesis:** Smaller models should generalize better (per Cho et al.'s finding). But too small may not have capacity for the base case. Expect a sweet spot around 1-2M params.

**Result:** PENDING

---

### EXP-055: Binary vs Decimal Representation

**Date:** PENDING
**Phase:** 5
**Config:**
- A: Binary (default)
- B: Decimal (each token = one digit 0-9)
- Same Karatsuba structure, same model size

**Hypothesis:** Binary should work better because splitting is cleaner. Decimal allows direct comparison with prior work.

**Result:** PENDING

---

### EXP-056: LoopFormer Shortcut-Consistency Training

**Date:** PENDING
**Phase:** 5
**Config:**
- A: Standard training (fixed loop count per example)
- B: LoopFormer-style training (varying loop depths with shortcut consistency loss)

**Hypothesis:** Shortcut consistency should improve generalization to deeper recursion (more loops at test time).

**Result:** PENDING

**Notes:** Based on LoopFormer (Jeddi et al., Feb 2026, arxiv 2602.11451).

---

### EXP-057: Auxiliary Task Co-Training (Addition)

**Date:** PENDING
**Phase:** 5
**Config:**
- A: Train on Karatsuba multiplication only
- B: Co-train on Karatsuba multiplication + binary addition (longer sequences)
- Same total training steps, 50/50 mix

**Hypothesis:** Per Cai et al. (NeurIPS 2025), length generalization can transfer from an easier task (addition, which generalizes well) to a harder task (multiplication). Co-training should improve multiplication generalization.

**Result:** PENDING

---

## Phase 6: Mechanistic Interpretability

### EXP-060: Attention Pattern Analysis

**Date:** PENDING
**Phase:** 6
**Config:**
- Model: Best from Phase 3-4
- Analyze attention patterns at each loop iteration
- Group by step type (SPLIT, MUL, ADD, SUB, COMBINE)

**Hypothesis:** Different attention heads should specialize for different operations. Different loop iterations should correspond to different recursion levels.

**Result:** PENDING

**Notes:** Look for:
- Head specialization (one head for splitting, one for carrying, etc.)
- Phase transition between decomposition and recombination iterations
- Whether attention patterns at loop t on 16-bit input match patterns at loop t on 8-bit input

---

### EXP-061: Fourier Analysis of Embeddings

**Date:** PENDING
**Phase:** 6
**Config:**
- Analyze learned embeddings of bit tokens (0 and 1) and position embeddings
- Apply DFT and look for structure (per Nanda et al.'s approach)

**Hypothesis:** The model may learn Fourier-like structure in the embeddings, even without explicit Fourier features. Binary representation might lead to interesting frequency-domain patterns.

**Result:** PENDING

---

### EXP-062: Loop Utilization Analysis

**Date:** PENDING
**Phase:** 6
**Config:**
- Track hidden state norms and directions at each loop iteration
- Compare 8-bit (training) vs 16-bit (generalization) vs 32-bit (deeper generalization)

**Hypothesis:** The model should use early loops for decomposition and later loops for recombination. On longer inputs, it should use additional loops for the extra recursion levels while keeping the base-case behavior the same.

**Result:** PENDING

**Notes:** Plot hidden state trajectory (PCA/t-SNE) across loop iterations. Look for the "recursion structure" in the latent space.

---

### EXP-063: Does the Model Actually Learn Karatsuba?

**Date:** PENDING
**Phase:** 6
**Config:**
- Intervene on hidden states at specific loop iterations
- Zero out or randomize specific directions
- Test whether the model's behavior matches Karatsuba's structure or something else

**Hypothesis:** The model may learn something mathematically equivalent to Karatsuba but implemented differently in the weights. Or it may learn a hybrid algorithm. Understanding what it actually learns is the most interesting scientific question.

**Result:** PENDING

**Notes:** This is what would make the paper publishable at a top venue.
