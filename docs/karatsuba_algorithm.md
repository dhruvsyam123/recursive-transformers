# The Karatsuba Algorithm: Detailed Reference

## Historical Context

The Karatsuba algorithm was discovered by Anatoly Karatsuba in 1960 and published in 1962. It was the first multiplication algorithm asymptotically faster than the quadratic "grade school" (long multiplication) algorithm. It demonstrated that multiplication could be performed in fewer than O(n^2) elementary operations, disproving the conjecture by Kolmogorov that this was impossible.

---

## The Algorithm

### Core Idea

To multiply two n-digit numbers X and Y, the naive approach requires 4 multiplications of n/2-digit numbers. Karatsuba's key insight is that this can be reduced to **3 multiplications** of n/2-digit numbers, plus some additions and subtractions (which are O(n)).

### Formal Description

Given two n-bit numbers X and Y, split each into high and low halves:

```
X = X_hi * B^(n/2) + X_lo
Y = Y_hi * B^(n/2) + Y_lo
```

where B is the base (B=2 for binary).

**Naive expansion** (4 multiplications):
```
X * Y = X_hi*Y_hi * B^n + (X_hi*Y_lo + X_lo*Y_hi) * B^(n/2) + X_lo*Y_lo
```

**Karatsuba's trick** (3 multiplications):

Compute three products:
```
z0 = X_lo * Y_lo
z2 = X_hi * Y_hi
z1 = (X_lo + X_hi) * (Y_lo + Y_hi) - z0 - z2
```

Then:
```
X * Y = z2 * B^n + z1 * B^(n/2) + z0
```

The key observation is that:
```
z1 = X_lo*Y_hi + X_hi*Y_lo
```
which is exactly the "cross term" we need, but computed using only **one** multiplication (of (X_lo + X_hi) and (Y_lo + Y_hi)) instead of two.

### Why 3 Multiplications Instead of 4

The naive approach computes: X_hi*Y_hi, X_hi*Y_lo, X_lo*Y_hi, X_lo*Y_lo (4 multiplications).

Karatsuba observes:
```
(X_lo + X_hi)(Y_lo + Y_hi) = X_lo*Y_lo + X_lo*Y_hi + X_hi*Y_lo + X_hi*Y_hi
```

So:
```
X_lo*Y_hi + X_hi*Y_lo = (X_lo + X_hi)(Y_lo + Y_hi) - X_lo*Y_lo - X_hi*Y_hi
```

We already computed X_lo*Y_lo (as z0) and X_hi*Y_hi (as z2), so the cross term comes "for free" from one additional multiplication minus two already-computed values.

---

## Concrete Binary Example: 8-bit x 8-bit

### Setup

Let us multiply two 8-bit numbers:
```
X = 10110011  (= 179 in decimal)
Y = 11010110  (= 214 in decimal)
```

Expected result: 179 * 214 = 38,306 = 1001010110000010 (16 bits)

### Step-by-step Walkthrough

**Step 1: Split into 4-bit halves**
```
X_hi = 1011  (= 11)      X_lo = 0011  (= 3)
Y_hi = 1101  (= 13)      Y_lo = 0110  (= 6)
```

Here B = 2, n = 8, so B^(n/2) = 2^4 = 16.

**Step 2: Compute z0 = X_lo * Y_lo**
```
z0 = 0011 * 0110 = 3 * 6 = 18 = 00010010
```

**Step 3: Compute z2 = X_hi * Y_hi**
```
z2 = 1011 * 1101 = 11 * 13 = 143 = 10001111
```

**Step 4: Compute the sums for z1**
```
X_lo + X_hi = 0011 + 1011 = 1110  (= 14)
Y_lo + Y_hi = 0110 + 1101 = 10011 (= 19)
```

Note: The sums may be (n/2 + 1) bits wide (one bit of carry). This is important for implementation.

**Step 5: Compute the product for z1**
```
(X_lo + X_hi) * (Y_lo + Y_hi) = 14 * 19 = 266 = 100001010
```

**Step 6: Compute z1**
```
z1 = 266 - z0 - z2 = 266 - 18 - 143 = 105 = 01101001
```

**Step 7: Combine**
```
Result = z2 * 2^8 + z1 * 2^4 + z0
       = 143 * 256 + 105 * 16 + 18
       = 36,608 + 1,680 + 18
       = 38,306
```

In binary (with shifts):
```
z2 << 8:  10001111 00000000     (z2 shifted left by 8)
z1 << 4:      01101001 0000     (z1 shifted left by 4)
z0:                00010010     (z0, no shift)

Sum:      1001010110000010      = 38,306  (correct!)
```

---

## Full Recursion Trace (8-bit with 2-bit base case)

For the transformer project, we use a 4-bit base case (or 2-bit for deeper recursion). Here is the full recursion trace with a **2-bit base case**, showing the tree structure:

### Multiply: X = 10110011 (179) by Y = 11010110 (214)

```
LEVEL 0: 10110011 * 11010110  (8-bit * 8-bit)
  Split: X_hi=1011, X_lo=0011, Y_hi=1101, Y_lo=0110

  LEVEL 1a: z0 = 0011 * 0110  (4-bit * 4-bit)
    Split: X_hi=00, X_lo=11, Y_hi=01, Y_lo=10

    LEVEL 2a: z0 = 11 * 10  (2-bit * 2-bit -> BASE CASE)
      = 3 * 2 = 6 = 0110

    LEVEL 2b: z2 = 00 * 01  (2-bit * 2-bit -> BASE CASE)
      = 0 * 1 = 0 = 0000

    LEVEL 2c: sums = (11+00)=11, (10+01)=11
              product = 11 * 11  (2-bit * 2-bit -> BASE CASE)
              = 3 * 3 = 9 = 1001
              z1 = 9 - 6 - 0 = 3 = 0011

    Combine: z2*2^4 + z1*2^2 + z0 = 0 + 12 + 6 = 18 = 00010010

  LEVEL 1b: z2 = 1011 * 1101  (4-bit * 4-bit)
    Split: X_hi=10, X_lo=11, Y_hi=11, Y_lo=01

    LEVEL 2d: z0 = 11 * 01  (2-bit * 2-bit -> BASE CASE)
      = 3 * 1 = 3 = 0011

    LEVEL 2e: z2 = 10 * 11  (2-bit * 2-bit -> BASE CASE)
      = 2 * 3 = 6 = 0110

    LEVEL 2f: sums = (11+10)=101, (01+11)=100
              product = 101 * 100  (3-bit * 3-bit)
              Note: sums can overflow! This needs careful handling.
              = 5 * 4 = 20 = 10100
              z1 = 20 - 3 - 6 = 11 = 1011

    Combine: z2*2^4 + z1*2^2 + z0 = 96 + 44 + 3 = 143 = 10001111

  LEVEL 1c: z1 computation
    X_lo + X_hi = 0011 + 1011 = 1110 (= 14)
    Y_lo + Y_hi = 0110 + 1101 = 10011 (= 19)

    Now multiply 1110 * 10011 (these are 4-5 bit numbers)
    This sub-problem is slightly larger than 4-bit due to carry!

    Split: treating as 5-bit numbers padded:
      01110 * 10011
      X_hi=01, X_lo=110, Y_hi=10, Y_lo=011
      (Or split evenly into ~3-bit halves)

    ... (recursion continues to base cases)

    Product = 14 * 19 = 266
    z1 = 266 - 18 - 143 = 105

  LEVEL 0 Combine:
    z2 * 2^8 + z1 * 2^4 + z0
    = 143 * 256 + 105 * 16 + 18
    = 36608 + 1680 + 18
    = 38306 = 1001010110000010
```

### Important Implementation Note: Carry Overflow

When computing (X_lo + X_hi) and (Y_lo + Y_hi), the sums can be one bit wider than the operands. For n/2-bit halves, the sums are at most (n/2 + 1) bits. This means the z1 sub-multiplication operates on slightly larger numbers than n/2 bits. This is a well-known complication of Karatsuba in practice, and for the transformer scratchpad, it needs to be handled carefully (e.g., by always padding to a consistent width).

---

## Depth-First Scratchpad Trace Format (for Transformer Training)

For an 8-bit * 8-bit multiplication with 4-bit base case (1 level of recursion):

```
[INPUT] 10110011 * 11010110
[SPLIT_8] X_hi=1011 X_lo=0011 Y_hi=1101 Y_lo=0110
[MUL_BASE] z0 = 0011 * 0110 = 00010010
[MUL_BASE] z2 = 1011 * 1101 = 10001111
[ADD] sum_X = 0011 + 1011 = 01110
[ADD] sum_Y = 0110 + 1101 = 10011
[MUL_BASE] prod = 01110 * 10011 = 100001010
[SUB] z1 = 100001010 - 00010010 - 10001111 = 01101001
[COMBINE] result = 10001111<<8 + 01101001<<4 + 00010010
[OUTPUT] 1001010110000010
```

For a 16-bit * 16-bit (2 levels of recursion), each [MUL_BASE] above would itself expand into a [SPLIT] -> [MUL_BASE]*3 -> [ADD] -> [SUB] -> [COMBINE] sub-trace, making the representation recursive.

---

## Complexity Analysis

### Time Complexity

The recurrence relation for Karatsuba is:
```
T(n) = 3 * T(n/2) + O(n)
```

- 3 recursive calls on problems of size n/2
- O(n) work for additions, subtractions, and shifts at each level

By the Master Theorem (Case 1: a=3, b=2, f(n)=O(n), log_b(a) = log_2(3) approx 1.585):
```
T(n) = O(n^(log_2 3)) = O(n^1.585)
```

### Comparison with School Multiplication

| Property | School Multiplication | Karatsuba |
|----------|----------------------|-----------|
| Time complexity | O(n^2) | O(n^1.585) |
| Sub-multiplications per level | 4 (or n) | 3 |
| Recursion depth | 1 (iterative) | log_2(n) |
| Additional operations | O(n) additions | O(n) add/sub per level |
| Total additions | O(n^2) | O(n^1.585) |
| Per-step complexity | O(n) (each partial product) | O(1) (bounded at each node) |

### Space Complexity

- The recursion depth is O(log n)
- At each level, we store O(n) intermediate values
- Total space: O(n * log n) for the full recursion trace
- For the scratchpad: the full depth-first trace has O(n^1.585) tokens total

### Number of Operations at Each Level

```
Level 0: 1 problem of size n        -> 1 * O(n) additions
Level 1: 3 problems of size n/2     -> 3 * O(n/2) additions
Level 2: 9 problems of size n/4     -> 9 * O(n/4) additions
...
Level k: 3^k problems of size n/2^k -> 3^k * O(n/2^k) additions
...
Level log_2(n): 3^(log_2 n) = n^(log_2 3) base cases
```

Total work across all levels:
```
Sum_{k=0}^{log_2(n)} 3^k * n/2^k = n * Sum_{k=0}^{log_2(n)} (3/2)^k = O(n^(log_2 3))
```

---

## How Karatsuba Maps to Transformer Operations

This is the key insight for our project. Each step of Karatsuba can be mapped to a bounded-complexity operation that a transformer can learn:

### Operation Breakdown

| Karatsuba Step | Transformer Operation | Complexity |
|---------------|----------------------|------------|
| **Split** | Route bits to two groups based on position | O(1) per token via position encoding |
| **Base case multiply** | 4-bit * 4-bit lookup (256 cases) | Learnable by a small transformer |
| **Addition** (for sums) | Binary addition with carry | O(n) but well-studied, transformers can learn this |
| **Subtraction** (for z1) | Binary subtraction with borrow | Similar to addition |
| **Shift** (B^(n/2) multiplication) | Reindex positions | O(1) via position encoding |
| **Combine** (shifted addition) | Multi-operand addition | O(n), straightforward |

### Why This Is Better Than School Multiplication for Transformers

**School multiplication** requires each output digit to attend to a linearly growing number of input digits (all partial products that contribute to that column). For n-digit multiplication, some output digits depend on O(n) input digits. As n grows, the attention pattern becomes increasingly complex.

**Karatsuba** keeps the per-step computation **bounded**:
- Splitting is O(1) per token (just look at position)
- Base case multiplication is on fixed-size inputs (e.g., 4 bits)
- Each addition/subtraction step involves at most 2 operands
- The model only needs O(log n) iterations (loop steps) to handle any input length

### Mapping to Looped Transformer Iterations

**Option A: One loop per recursion level (top-down, then bottom-up)**
```
Loop 1 (top-down):   Split n-bit into n/2-bit halves, compute sums
Loop 2 (top-down):   Split n/2-bit into n/4-bit halves, compute sums
...
Loop log(n) (base):  Multiply base-case-sized numbers
Loop log(n)+1 (up):  Combine results at deepest level
...
Loop 2*log(n) (up):  Combine results at top level, produce output
```
Total: ~2 * log_2(n) loops

**Option B: One loop per trace step**
```
Loop 1: Split
Loop 2: Begin z0 sub-problem (split again)
Loop 3: z0's z0 base case multiply
Loop 4: z0's z2 base case multiply
...
```
Total: ~O(n^0.585) loops (proportional to number of nodes in the recursion tree)

### Position Encoding for Karatsuba

Each token needs to encode:
1. **Bit significance** (which bit position, 0 = LSB): Tells the model which half a bit belongs to after splitting
2. **Recursion depth** (0 = top, log(n) = base case): Tells the model how deep in the recursion tree
3. **Sub-problem ID** (which of the 3 sub-problems: z0, z1, z2): Routes information correctly
4. **Step type** (SPLIT, MUL, ADD, SUB, COMBINE): What operation to perform

This 4-tuple position encoding captures the full recursive structure and enables the transformer to generalize: for a longer input, the model simply adds more recursion levels, using the same operations at each level.

---

## Comparison: School vs. Karatsuba for a 32-bit Multiplication

### School Algorithm
- Produces 32 partial products (each 32 bits shifted)
- Adds them all up (32 x 32 = 1024 single-digit multiplications)
- Each output bit depends on up to 32 input bits
- Total operations: O(n^2) = O(1024)

### Karatsuba Algorithm
- Level 0: 1 split, 2 additions -> 3 sub-problems of 16 bits
- Level 1: 3 splits, 6 additions -> 9 sub-problems of 8 bits
- Level 2: 9 splits, 18 additions -> 27 sub-problems of 4 bits
- Level 3 (base): 27 4-bit multiplications
- Combine levels: 27 + 9 + 3 + 1 = 40 combine steps
- Total multiplications at base: 27 (vs. 1024 for school)
- Recursion depth: 3 (with 4-bit base case)
- Each step involves bounded-size operands

### Scratchpad Length Comparison (Approximate)

For n-bit * n-bit multiplication:
- School scratchpad: O(n^2) tokens (all partial products)
- Karatsuba scratchpad (depth-first): O(n^1.585) tokens
- Karatsuba scratchpad (with 4-bit base case, starting from 8 bits): very compact

| Input size | School scratchpad tokens | Karatsuba scratchpad tokens | Karatsuba recursion depth |
|-----------|-------------------------|---------------------------|--------------------------|
| 8-bit     | ~100                    | ~50                       | 1 (with 4-bit base)     |
| 16-bit    | ~400                    | ~150                      | 2                        |
| 32-bit    | ~1600                   | ~450                      | 3                        |
| 64-bit    | ~6400                   | ~1350                     | 4                        |
| 128-bit   | ~25600                  | ~4050                     | 5                        |

The advantage grows with input size, which is exactly the regime where length generalization matters.

---

## Implementation Notes for the Transformer Project

1. **Use binary representation**: Karatsuba is cleanest in binary. Splitting is simply taking the top and bottom n/2 bits. No messy carry handling during the split itself.

2. **Handle carry overflow in sums**: When computing X_lo + X_hi, the result can be (n/2 + 1) bits. The scratchpad format must accommodate this. One approach: always pad to a fixed width.

3. **4-bit base case is optimal for starting**: It gives 256 possible products (easily memorizable), and means 8-bit training data only needs 1 recursion level. Testing on 16-bit needs 2 levels, 32-bit needs 3 levels, etc.

4. **Depth-first ordering for scratchpad**: Process each sub-problem fully before moving to the next. This keeps the "working memory" (context window) requirements bounded, as each sub-problem's result is fully computed before it is needed.

5. **Intermediate supervision**: Add loss not just on the final output, but on each intermediate result in the trace. This prevents error accumulation and helps the model learn each step independently.
