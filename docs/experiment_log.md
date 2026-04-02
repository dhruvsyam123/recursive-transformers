# Experiment Log

---

## Phase 1: 4-bit Base Case

### EXP-001: 4-bit Base Case with Sinusoidal Positions

**Date:** 2026-03-30
**Config:**
- Model: 1 shared layer, 4 heads, d=128, 4 loops, sinusoidal positions
- Data: 4-bit × 4-bit, all 256 pairs, train_fraction=1.0
- Training: AdamW, lr=3e-3 cosine, weight_decay=0.01, batch_size=32, 500 epochs
- Hardware: T4 GPU (Colab)

**Bug Found:** Off-by-one in `make_predictable_mask`. Mask computed on `input_ids` but applied to `target_ids`. Position 0 (input=[INPUT], target=x0) marked as predictable when x0 is random.

**Result: FAILURE → 50.0% exact (128/256)**
- Loss plateaued at 0.0365 (matches theoretical: 1 impossible token per 18 predictable = -log(0.5)/18 ≈ 0.038)
- Exactly 50% = all pairs where x is even (model always predicts x0=0)

**Fix:** Compute mask on full `token_ids`, shift by 1 (`full_mask[:, 1:]`).

### EXP-002: 4-bit Base Case with Fixed Mask

**Date:** 2026-03-30
**Config:** Same as EXP-001 but with fixed mask, 1000 epochs, decay_steps=8000

**Result: SUCCESS → 100% (256/256)**
- Loss reached 0.0000 at epoch 600
- All 256 pairs correct

---

## Phase 2: 8-bit Karatsuba (Sinusoidal Positions)

### EXP-003: 8-bit with Sinusoidal Positions

**Date:** 2026-03-30
**Config:**
- Model: 1 shared layer, 8 heads, d=256, 8 loops, sinusoidal positions
- Data: 3200 8-bit + 204 4-bit, DFS ordering, max_len=400
- Training: AdamW, lr=1e-3 cosine, weight_decay=0.01, batch_size=32
- Hardware: T4 GPU

**Result: PARTIAL → 93-97% on 8-bit, 0% on 16-bit**
- ~100 total epochs across multiple restarts with warm restarts
- Loss ~0.0004
- 16-bit generalization: 0/200 = 0.0%
- Model memorized 8-bit patterns, did not learn reusable procedure

---

## Phase 3: Hierarchical Position Encoding Experiments

### EXP-004: Hierarchical Positions Only (Sum Mode)

**Date:** 2026-03-30
**Config:**
- Model: 1 shared layer, 8 heads, d=256, 6 loops, hierarchical position encoding (sum mode)
- Bug fix: num_step_types=10 (was 7, causing COMBINE/OUTPUT/BASE_MUL OOB)
- Data: 4-bit + 8-bit, DFS ordering
- Training: weight_decay=0.01

**Result: FAILURE → 18% on 8-bit after 200 epochs**
- Removing sequential position info cripples autoregressive prediction
- Model can't distinguish token ordering

### EXP-005: Combined Sinusoidal + Hierarchical

**Date:** 2026-03-30
**Config:**
- CombinedModel: sinusoidal base model + separate HierarchicalPositionEncoding
- 1.35M parameters

**Result: FAILURE → 16% on 8-bit after 160 epochs**
- Hierarchical noise at init disrupts sinusoidal signal
- Loss plateau at 0.0075

### EXP-006: Hierarchical CONCAT — All Bugs Fixed ★

**Date:** 2026-03-30
**Config:**
- KaratsubaModel with HierarchicalPositionEncoding in CONCAT mode
- num_step_types=10, tag tokens bit_sig=200+step_type
- Weight decay 0.15 (!!), randomized loops [4,6,8]
- Separate 4-bit (bs=64, ml=29) and 8-bit (bs=16, ml=373) batches
- 759K parameters
- Hardware: A100 GPU

**Result: PARTIAL → 98.5% on 8-bit (grokking!), 0% on 16-bit**
- Classic grokking trajectory: 14% → 45% → 80% → 90% → 98.5% over 250 epochs
- Weight decay 0.15 was key to grokking
- 16-bit: 0/200 = 0.0%
- Root cause: learned embeddings for recursion_depth and sub_problem_id return random vectors for unseen values (depth=2, sub_id>3)

---

## Phase 4: NoPE + Input Injection + BFS

### EXP-007: NoPE + Input Injection + BFS + Grokfast

**Date:** 2026-04-02
**Config:**
- KaratsubaModel: no positional encoding, input injection (scale=0.1), sinusoidal timestep
- Breadth-first trace ordering
- Grokfast EMA (alpha=0.98, lambda=2.0)
- Weight decay 0.15
- Hardware: A100 GPU

**Result: FAILURE → 0% on 8-bit after 250 epochs**
- Loss dropped to 0.037 but accuracy stuck at 0%
- NoPE can't distinguish identical bit tokens at different positions in 300+ token sequences
- Changed two variables at once (NoPE + BFS) — can't tell which caused failure
- Grokfast may have been counterproductive

**Lessons:**
1. Never change two variables at once
2. NoPE doesn't work for long structured traces
3. Grokfast not needed (wasn't used in EXP-006 which got 98%)

---

## Phase 5: All-Sinusoidal Positions + Input Injection

### EXP-008: All-Sinusoidal Hierarchical + Input Injection ★

**Date:** 2026-04-02
**Config:**
- AllSinusoidalHierarchicalPos: all 4 components sinusoidal (not learned), concat mode
- LoopBlock with sinusoidal timestep (not learned)
- Input injection at each loop (scale=0.1, learnable)
- DFS ordering, weight_decay=0.15, randomized loops [4,6,8,10,12]
- Separate batches, 500 epochs
- Hardware: A100 GPU

**Result: PARTIAL → 99% on 8-bit (epoch 200), 0% on 16-bit**
- Faster grokking than EXP-006: 37% at epoch 50, 99% at epoch 200
- All-sinusoidal learns faster than learned embeddings
- 16-bit: 0/200 = 0.0%
- Confirmed: position encoding is necessary but not sufficient for generalization
- The fundamental barrier is attention OOD (1000+ tokens vs 370 training) + autoregressive cascade

---

## Phase 6: Recursive Neural Karatsuba (Option A)

### EXP-009: Teacher-Forced Recursive Evaluator ★★

**Date:** 2026-04-02
**Config:**
- Model from EXP-008 (99% on 8-bit)
- Classical Karatsuba recursion with model_multiply() as base case (≤8-bit)
- Teacher forcing: model sees correct trace, predicts product bits
- Tested on 8, 16, 32, 64-bit

**Result: SUCCESS**

| Bits | Examples | Model Calls/Example | Accuracy | Time |
|------|----------|--------------------:|----------|------|
| 8 | 100 | 1 | 100% | ~6 min |
| 16 | 200 | 5 | 100% | ~28 min |
| 32 | 50 | 17 | 100% | ~25 min |
| 64 | 20 | 55 | pending | pending |

**Comparison:**

| Method | 8-bit | 16-bit | 32-bit |
|--------|-------|--------|--------|
| Flat sequence (EXP-008) | 99% | 0% | N/A |
| Recursive evaluator | 100% | 100% | 100% |

**Significance:**
- Model trained on 4-bit + 8-bit ONLY, never saw 16-bit or 32-bit
- Each recursive call is bounded-length (≤370 tokens), within training distribution
- The model learned a useful multiplication subroutine that composes correctly
- Teacher forcing means model verifies correct traces rather than generating independently

---

## Phase 7: PAUSE/RESUME Recursive Self-Invocation

### EXP-010: PAUSE/RESUME Single-Level Traces (PENDING)

**Date:** 2026-04-02
**Config:**
- Model: Same as Phase 5 (all-sinusoidal + input injection) but VOCAB_SIZE=145 (+[PAUSE], [RESUME])
- Data: 4-bit base cases (256 pairs, 27 tokens) + 8-bit single-level traces (4000 pairs, ~157 tokens)
- Trace format: Sub-problems replaced with [PAUSE]/[RESUME] instead of inlined
- Training: teacher-forced, weight_decay=0.15, randomized loops [4,6,8]
- Inference: autoregressive with recursive self-invocation (model emits [PAUSE], runtime recurses)
- Each model call bounded at ~157 tokens regardless of total input bit width
- Hardware: A100 GPU

**Hypothesis:** The model learns when to recurse (emit [PAUSE]) and how to combine results (after [RESUME]). At inference, the runtime manages the call stack. Since each call is bounded-length and structurally identical to training, 16-bit should generalize.

**Key difference from Phase 6:** The model generates tokens AUTOREGRESSIVELY (not teacher-forced). It decides when to PAUSE, outputs sub-problem operands, and computes SUB/COMBINE from RESUME results. The recursion control is partially learned.

**Result:** PENDING

---

## Key Findings Summary

### Bugs Found and Fixed
1. **Off-by-one mask** (EXP-001): Causes exact 50% accuracy. Fix: compute on full tokens, shift by 1.
2. **num_step_types=7** (EXP-004): Karatsuba has 10 step types. COMBINE/OUTPUT/BASE_MUL got garbage embeddings.
3. **Tag token bit_sig=-1** (EXP-006): Collides with LSB tokens. Fix: map to 200+step_type.
4. **Infinite recursion** in recursive evaluator: next_pow2 rounded sub-problem sizes back up. Fix: use actual sizes.

### What Works
- Weight decay 0.15 → grokking (algorithm learning, not memorization)
- Hierarchical concat positions (4 × d_model/4 dedicated dims)
- All-sinusoidal > learned embeddings (faster convergence)
- DFS trace ordering
- Separate batching by bit width (no padding waste)
- Input injection (marginal improvement)
- Recursive evaluator with neural subroutine → 100% at 16-bit and 32-bit

### What Doesn't Work
- Sinusoidal-only positions for generalization (memorizes, 0% OOD)
- Learned embeddings for depth/sub_problem_id (OOD for unseen values)
- NoPE for 300+ token sequences
- BFS ordering (untested in isolation, failed with NoPE)
- Grokfast (unnecessary, possibly harmful)
- Flat-sequence processing for length generalization on N×N multiplication

### Why Flat-Sequence Length Generalization Fails
1. **Autoregressive cascade:** 0.99^800 ≈ 0% sequence accuracy for 16-bit traces
2. **Attention OOD:** 1000+ token sequences vs 370 max during training
3. **Compositional gap:** Model saw 1 recursion level, needs to compose 2
4. **Variable-width combine:** COMBINE step handles 32-bit numbers at 16-bit vs 16-bit at 8-bit training — per-step dependency count grows with problem size
