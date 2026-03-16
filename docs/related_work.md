# Related Work: Detailed Paper Summaries

This document provides detailed summaries of the most important papers for the Karatsuba-style recursive multiplication project. Papers are organized by relevance.

---

## 1. Hou et al. (ICML 2025) -- "Universal Length Generalization with Turing Programs"

**ArXiv:** [2407.03310](https://arxiv.org/abs/2407.03310)
**Authors:** Kaiying Hou, David Brandfonbrener, Sham M. Kakade, Samy Jelassi, Eran Malach
**Venue:** ICML 2025

### Key Idea

Proposes **Turing Programs**, a chain-of-thought (CoT) strategy that decomposes any algorithmic task into steps mimicking the computation of a Turing Machine. The approach is **universal** (works for any computable function) and **simple** (each step is essentially copying the previous tape state with a few local modifications).

### Method

1. **Turing Machine Simulation as Scratchpad**: Any algorithm can be described as a Turing Machine. The scratchpad represents the TM tape at each timestep. At each step, the transformer produces a new copy of the tape with a few local modifications (the head moves, one cell is written).

2. **Hard-ALiBi Positional Encoding**: A variant of ALiBi (Attention with Linear Biases) where the attention bias is "hard" -- it constrains each token to only attend to tokens within a fixed window of m recent positions. This is critical because:
   - It makes each generation step depend only on a local window, not the full context
   - The same local pattern repeats regardless of tape length
   - This enables length generalization: the transformer learns local copy-and-modify rules that work for any tape length

3. **Each step is a "modified copy"**: The fundamental operation at each step is copying the previous tape and making O(1) local changes. This is simple enough for a small transformer to learn and generalizes because the copying pattern is translation-invariant.

### Results

- **Addition**: 98% accuracy on 100-digit addition (trained on up to 50-digit), achieving 2x length generalization
- **Multiplication (n x 1)**: 97% accuracy when generalizing from 50-digit training to 100-digit testing
- **Multiplication (n x 3)**: 97% accuracy with the same 2x generalization factor
- **In-context SGD**: Also demonstrated length generalization on SGD simulation steps
- These were the **first results showing non-trivial length generalization on multiplication**

### How It Relates to Our Project

**This is our primary baseline and comparison target.** Key differences:
- Hou et al. use a **flat TM simulation** with O(n^2) total steps for multiplication (because the TM for multiplication takes O(n^2) steps)
- Our approach uses **recursive Karatsuba decomposition** with O(n^1.585) total work and O(log n) recursion depth
- Even matching their accuracy with fewer computational steps would be a significant contribution
- Their Hard-ALiBi insight (local attention enables length generalization) is complementary to our position coupling approach
- They focus on n x 1 and n x 3 multiplication; we target **n x n** multiplication, which is harder

### Limitations

- Requires O(n^2) scratchpad steps for n x n multiplication (because the underlying TM is quadratic)
- The scratchpad is very long -- each step copies the entire tape
- Does not exploit the mathematical structure of multiplication (treats it as a generic computation)

---

## 2. Cho et al. (NeurIPS 2024) -- "Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure"

**ArXiv:** [2405.20671](https://arxiv.org/abs/2405.20671)
**Authors:** Hanseul Cho, Jaeyoung Cha, Pranjal Awasthi, Srinadh Bhojanapalli, Anupam Gupta, Chulhee Yun
**Venue:** NeurIPS 2024
**Code:** [github.com/HanseulJo/position-coupling](https://github.com/HanseulJo/position-coupling)

### Key Idea

Instead of assigning unique position IDs to each token (standard absolute positional encoding), **position coupling** assigns the **same position ID** to tokens that are "structurally relevant" to each other. For addition, digits of the same significance (same place value) in the two operands and the result all share the same position ID.

### Method

1. **Position Assignment Rule**: For an addition problem "A + B = C", the ones digit of A, the ones digit of B, and the ones digit of C all get position ID 0. The tens digits all get position ID 1, etc.

2. **Theoretical Guarantee**: The authors prove that a **1-layer Transformer** with coupled positions can solve addition for exponentially many digits. Without position coupling, no 1-layer Transformer can solve addition for all lengths. This is a strong theoretical result showing that position coupling doesn't just help empirically -- it fundamentally changes what the model can express.

3. **Implementation**: Position coupling modifies only the positional encoding layer. The rest of the transformer architecture is unchanged. It uses learned absolute positional embeddings but with the coupled (shared) position IDs.

### Results

- **Addition**: Models trained on 1-30 digit additions generalize to **200-digit additions** (6.67x generalization) with >95% exact-match accuracy
- **N x 2 Multiplication**: Position coupling can be applied but results are limited to small operand counts
- **Sorting and other tasks**: Also demonstrated on sorting and a 2D grid navigation task
- The key result is the dramatic improvement from ~1x to 6.67x generalization on addition

### How It Relates to Our Project

**Position coupling is a foundation technique we build upon.** Our hierarchical position encoding extends position coupling to the recursive structure of Karatsuba:
- Cho et al. couple by digit significance (place value)
- We couple by digit significance AND recursion depth AND sub-problem ID AND step type
- Their approach handles the "which digits interact" problem; ours additionally handles the "which recursion level" problem
- Their theoretical result (1-layer can solve addition with coupling) suggests that coupling can make fundamentally hard problems tractable -- we aim for a similar result with recursive multiplication
- Their code is available and provides a starting point for implementation

### Limitations

- Only demonstrated on addition (not n x n multiplication)
- N x 2 multiplication results are limited
- Does not address the recursive structure needed for general multiplication
- Follow-up work (Cho et al., ICLR 2025, arxiv 2410.15787) extends to multiplication with scratchpads but still uses the school algorithm, not recursive decomposition

---

## 3. Fan et al. (ICLR 2025) -- "Looped Transformers for Length Generalization"

**ArXiv:** [2409.15647](https://arxiv.org/abs/2409.15647)
**Authors:** Ying Fan, Yilun Du, Kannan Ramchandran, Kangwook Lee
**Venue:** ICLR 2025
**Code:** [github.com/UW-Madison-Lee-Lab/looped-tf](https://github.com/UW-Madison-Lee-Lab/looped-tf)

### Key Idea

Weight-shared (looped) transformers with an **adaptive number of loop iterations** significantly improve length generalization on tasks that have iterative algorithmic solutions. The number of loops can be scaled at test time to handle longer inputs.

### Method

1. **RASP-L Framework**: The authors define RASP-L as a class of operations that are length-generalizable and expressible by a finite-sized transformer. Tasks that can be decomposed into n iterations of a RASP-L operation (called "n-RASP-L problems") are natural targets for looped transformers.

2. **Looped Architecture**: A single transformer block (or small stack of blocks) is applied repeatedly with shared weights. The key innovation is **step-dependent supervision**: the model is trained with a loss that depends on the loop iteration, enabling it to learn different behaviors at different iterations.

3. **Adaptive Loop Count**: At test time, the number of loops can be increased proportionally to the input length. For a task requiring n iterations on length-n inputs, the model uses n loops. This is the mechanism that enables length generalization: train with k loops on length-k inputs, test with 2k loops on length-2k inputs.

4. **Training Algorithm**: The paper proposes a specific training procedure:
   - Train with varying loop counts during training
   - Use step-dependent loss (different targets for different loop iterations)
   - This teaches the model to make incremental progress at each iteration

### Results

- **Parity**: Trained on up to 20-digit inputs, generalizes near-perfectly to 50+ digits
- **Addition**: Looped model generalizes to lengths well beyond training, where fixed-depth baselines fail completely
- **Copy**: Similarly strong generalization
- Vanilla next-token prediction (NTP) fails when tested on inputs just 10 tokens longer than training maximum
- The looped model with adaptive depth **significantly outperforms** vanilla NTP, NTP with pause tokens, and weight-tied layers without adaptive depth

### How It Relates to Our Project

**This is the architecture backbone for our project.** Key connections:
- Our looped transformer will use weight-shared blocks, following Fan et al.'s architecture
- The adaptive loop count maps directly to our recursion depth: more loops = more recursion levels = longer inputs
- Their RASP-L framework provides theoretical grounding: if each Karatsuba step is a RASP-L operation, then the looped architecture can learn the full recursion
- They did NOT attempt multiplication, leaving it as an open problem. We fill this gap.
- Their step-dependent supervision idea is relevant to our intermediate supervision (loss on each recursion level)

### Limitations

- Only tested on iterative tasks (addition, parity, copy) -- not recursive tasks like multiplication
- The RASP-L framework assumes the task decomposes into identical iterations, which is not quite true for Karatsuba (the decomposition and recombination phases are different)
- No mechanism for handling the tree structure of recursive computations (only linear iteration)

---

## 4. Saunshi et al. (ICLR 2025) -- "Reasoning with Latent Thoughts: On the Power of Looped Transformers"

**ArXiv:** [2502.17416](https://arxiv.org/abs/2502.17416)
**Authors:** Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, Sashank J. Reddi
**Venue:** ICLR 2025

### Key Idea

Many reasoning problems require large **depth** but not necessarily many **parameters**. A k-layer transformer looped L times nearly matches a kL-layer non-looped transformer, and looped models implicitly generate **"latent thoughts"** -- hidden-state representations that function like an internal chain of thought without explicit token generation.

### Theoretical Results

1. **Depth Equivalence**: A k-layer transformer looped L times is nearly as powerful as a kL-layer (non-looped) transformer for many reasoning tasks. This means we can get deep reasoning with a small model by looping.

2. **Latent Thought Simulation**: The paper proves that looped models can simulate T steps of explicit chain-of-thought (CoT) reasoning using T loop iterations, but **without generating intermediate tokens**. The hidden states carry the "thoughts" implicitly.

3. **Scaling Behavior**: Looped and non-looped models exhibit scaling behavior that depends on their **effective depth** (k * L for looped models). This is analogous to inference-time scaling of CoT reasoning.

4. **Practical Language Modeling**: The benefits translate to real language modeling: a k-layer model looped L times can be competitive with or better than a kL-layer model on downstream reasoning tasks.

### Empirical Results

- **Synthetic reasoning**: On tasks like addition, p-hop induction, and math problems, looped models with sufficient iterations match much deeper non-looped models
- **Language modeling**: Practical benefits observed on reasoning benchmarks
- The key finding is that effective depth (layers * loops) is what matters, not raw layer count

### How It Relates to Our Project

**This paper provides the theoretical validation for our entire approach.** Critical connections:
- Our hypothesis is that a looped transformer can learn Karatsuba recursion by using each loop as a recursion step. Saunshi et al. prove this is theoretically possible: T loops can simulate T steps of reasoning.
- The "latent thoughts" concept maps to our recursion trace: each loop iteration processes one level of the Karatsuba tree, with the hidden states carrying intermediate sub-problem results.
- Their depth-equivalence result means our small looped model (e.g., 2 layers, 10 loops) is as powerful as a 20-layer model, which should be more than enough for Karatsuba.
- The scaling behavior prediction means we should see performance improve smoothly as we increase loop count, which is testable in our experiments.
- Their work justifies using loops for recursive computation without explicit scratchpad generation at each step (Option C from our research plan).

### Limitations

- Theoretical results are for synthetic settings; real-world transfer is demonstrated but less rigorous
- Does not specifically address recursive (tree-structured) computations -- focuses on sequential reasoning chains
- Does not address position encoding or how the model knows which iteration/depth it is at (though timestep encoding is discussed)

---

## 5. Nanda et al. (ICLR 2023) -- "Progress Measures for Grokking via Mechanistic Interpretability"

**ArXiv:** [2301.05217](https://arxiv.org/abs/2301.05217)
**Authors:** Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt
**Venue:** ICLR 2023 (Oral)
**Blog:** [neelnanda.io/grokking-paper](https://www.neelnanda.io/grokking-paper)

### Key Idea

The authors **fully reverse-engineered** the algorithm learned by a small transformer trained on modular addition (a mod p + b mod p). The model learns to use **discrete Fourier transforms** and **trigonometric identities** to convert addition into rotation on a circle. This provides a concrete example of mechanistic interpretability for arithmetic.

### The Algorithm the Model Learns

The transformer does NOT learn "addition rules" in the way humans think about them. Instead:

1. **Embedding**: The model embeds input numbers a and b as points on a circle using Fourier features: cos(2*pi*k*a/p) and sin(2*pi*k*a/p) for key frequencies k.

2. **Attention**: The attention mechanism composes the rotations, effectively computing cos(2*pi*k*(a+b)/p) and sin(2*pi*k*(a+b)/p) using the trigonometric addition formulas:
   - cos(alpha + beta) = cos(alpha)*cos(beta) - sin(alpha)*sin(beta)
   - sin(alpha + beta) = sin(alpha)*cos(beta) + cos(alpha)*sin(beta)

3. **MLP/Output**: The output layer reads off the result from the combined rotation, mapping back from Fourier space to the answer.

This is called the **"Fourier multiplication" algorithm** (despite being used for addition) because it uses multiplication of Fourier coefficients.

### Three Phases of Training (Grokking)

The paper identifies three continuous phases:

1. **Memorization (epochs 0-1.4k)**: The model memorizes training examples. Training loss drops, test loss stays high. The model uses a high-weight, data-specific solution.

2. **Circuit Formation (epochs 1.4k-9.4k)**: The Fourier multiplication circuit gradually forms in the weights. The restricted loss (only using key frequencies) starts to drop. But test loss stays high because the memorization circuit still dominates.

3. **Cleanup (epochs 9.4k-14k)**: Weight decay gradually removes the memorization circuit. The Fourier circuit is now the dominant mechanism. Test loss suddenly drops -- this is the "grokking" moment. The sum of squared weights drops sharply.

### Progress Measures

The authors define two progress measures:
- **Restricted loss**: Performance when only key Fourier frequencies are kept (measures how well the generalizing circuit works)
- **Excluded loss**: Performance when key Fourier frequencies are ablated (measures how much the model relies on memorization)

These measures show continuous progress even during the "plateau" before grokking appears in test accuracy.

### The Role of Weight Decay

Weight decay is essential for grokking. It provides the pressure that eventually removes the memorization circuit in favor of the more parameter-efficient Fourier circuit. Without weight decay, the model never transitions from memorization to generalization.

### How It Relates to Our Project

This paper is relevant in several ways:

1. **Weight decay is critical**: For our Karatsuba transformer, weight decay will help prevent memorization and encourage the model to learn the recursive algorithm rather than memorize specific multiplication results. This is especially important given our small training set (65K examples of 8-bit multiplication).

2. **The model may not learn what we expect**: The modular addition model didn't learn "addition" -- it learned Fourier transforms. Our Karatsuba model might learn something mathematically equivalent but structurally different from Karatsuba. Mechanistic interpretability (Phase 6) will reveal what it actually learns.

3. **Progress measures for monitoring training**: We can define analogous progress measures:
   - Per-recursion-level accuracy (does the model get the base cases right before the combines?)
   - Attention pattern analysis (do attention heads specialize for different Karatsuba operations?)
   - These can detect learning before test accuracy improves

4. **Grokking is expected**: Our model will likely memorize the training set before generalizing. This paper tells us to be patient and to track the right metrics. Training for much longer than convergence on training loss may be necessary.

5. **Interpretability methodology**: The reverse-engineering approach (identify key features, define progress measures, track training dynamics) provides a template for Phase 6 of our project.

### Limitations

- Only studied modular addition on a very small transformer
- The Fourier mechanism is specific to modular arithmetic; it may not apply to general multiplication
- The model studied is tiny (1-layer, small embedding) -- unclear how findings scale

---

## Additional Important Papers (Brief Summaries)

### Cho et al. (ICLR 2025) -- "Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count"
**ArXiv:** [2410.15787](https://arxiv.org/abs/2410.15787)

Extension of position coupling to multiplication with scratchpads. Achieves ~2-3x length generalization on multiplication using multi-level position coupling. Uses the school algorithm (not recursive). The most directly comparable prior result for multiplication length generalization with position coupling.

### Jelassi et al. (2023) -- "Length Generalization in Arithmetic Transformers"

Early work showing that relative position embeddings fail for multiplication. Proposed train-set priming as a technique. Achieved 5-digit x 3-digit to 35 x 3 generalization, but only with one operand length fixed. Established the difficulty of multiplication length generalization.

### Giannou et al. (ICML 2023) -- "Looped Transformers as Programmable Computers"

Proves theoretically that looped transformers can emulate basic computational primitives (registers, conditional branching, memory access). Provides the theoretical foundation for why our approach should work: if looped transformers are Turing-complete, they can certainly implement Karatsuba.

### Dehghani et al. (2018) -- "Universal Transformers"

The original weight-sharing/looped transformer paper. Introduced adaptive computation time (ACT) for deciding when to stop iterating. Our architecture builds on this foundation.

### McLeish et al. (NeurIPS 2024) -- "Transformers Can Do Arithmetic with the Right Embeddings"
**ArXiv:** [2405.17399](https://arxiv.org/abs/2405.17399)

Per-digit position embeddings achieve 99% accuracy on 100-digit addition (trained on 20-digit). Also improves multiplication. Confirms that position encoding design is critical for arithmetic generalization.

### Izzo et al. (Oct 2025) -- "Quantitative Bounds for Length Generalization in Transformers"
**ArXiv:** [2510.27015](https://arxiv.org/abs/2510.27015)

First quantitative bounds on required training length for generalization. Proves length generalization occurs when internal behavior on longer sequences can be "simulated" by shorter ones. **Directly supports our recursive decomposition**: each level of Karatsuba reduces to shorter sub-problems, satisfying their "simulability" condition.

### Sato et al. (ICML 2025 Workshop) -- "Chain of Thought in Order"
**ArXiv:** [2506.23875](https://arxiv.org/abs/2506.23875)

Order of reasoning steps critically affects difficulty. Proposes hierarchical approach to discover optimal orderings. For multiplication, recovers reverse-digit order. **Relevant to our scratchpad format** -- the order in which we present the Karatsuba trace (depth-first vs breadth-first) matters for learnability.

### Cai et al. (NeurIPS 2025) -- "Extrapolation by Association"
**ArXiv:** [2506.09251](https://arxiv.org/abs/2506.09251)

Length generalization can transfer across related tasks. Training with a longer auxiliary task enables generalization on a shorter target task. **Potential technique**: co-train with longer addition sequences to bootstrap multiplication generalization.

### LoopFormer -- Jeddi et al. (Feb 2026)
**ArXiv:** [2602.11451](https://arxiv.org/abs/2602.11451)

Introduces shortcut-consistency training that aligns trajectories of different loop lengths. Conditions each loop step on internal time t and step size dt. **Directly relevant** to our adaptive loop count problem: helps the model handle different recursion depths robustly.

### SpiralFormer -- Yu, Shu et al. (Feb 2026)
**ArXiv:** [2602.11698](https://arxiv.org/abs/2602.11698)

Multi-resolution recursion schedule: early iterations capture global interactions on compressed sequences, later iterations refine at token resolution. **Mirrors our hierarchical recursion levels**: early loops handle top-level splits, later loops handle base cases.

---

## Summary Table: Key Results Comparison

| Paper | Task | Train Length | Test Length | Generalization Factor | Accuracy |
|-------|------|-------------|------------|----------------------|----------|
| Hou et al. (ICML 2025) | n x 1 mult | 50-digit | 100-digit | 2x | 97% |
| Hou et al. (ICML 2025) | n x 3 mult | 50-digit | 100-digit | 2x | 97% |
| Hou et al. (ICML 2025) | Addition | 50-digit | 100-digit | 2x | 98% |
| Cho et al. (NeurIPS 2024) | Addition | 30-digit | 200-digit | 6.67x | >95% |
| Cho et al. (ICLR 2025) | N x N mult | ~10-digit | ~20-30 digit | ~2-3x | varies |
| Fan et al. (ICLR 2025) | Parity | 20-digit | 50+ digit | 2.5x+ | ~100% |
| Fan et al. (ICLR 2025) | Addition | varies | much longer | significant | ~100% |
| McLeish et al. (NeurIPS 2024) | Addition | 20-digit | 100-digit | 5x | 99% |
| **Our target** | **N x N mult** | **8-bit** | **32-128 bit** | **4-16x** | **>90%** |

---

## Key Takeaways for Our Project

1. **Position encoding matters enormously** (Cho, McLeish). Our hierarchical position encoding is a critical design decision.

2. **Looped architectures enable length generalization** (Fan, Saunshi). Using loops to map to recursion levels is well-motivated.

3. **The scratchpad format matters** (Hou, Sato). Our depth-first Karatsuba trace needs to be carefully designed for learnability.

4. **Weight decay is essential** (Nanda). Without it, the model will memorize rather than learn the algorithm.

5. **Local attention helps** (Hou, Hard-ALiBi). The model should not need to attend to the full context at each step.

6. **No one has tried recursive decomposition for multiplication**. This is our novel contribution. The closest work (Hou et al.) uses flat TM simulation, and all position-coupling work uses the school algorithm.

7. **The theoretical foundation is strong** (Saunshi, Izzo). Looped transformers can simulate CoT, and recursive decomposition satisfies the conditions for length generalization.
