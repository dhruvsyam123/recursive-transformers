"""
School (grade-school / long) multiplication trace generator.

Generates structured traces of the standard long multiplication algorithm
in binary representation. This serves as the baseline comparison against
the Karatsuba trace.

In binary, long multiplication is:
  1. For each bit of the second operand that is 1, create a shifted copy
     of the first operand (a partial product).
  2. Sum all partial products.

The trace format uses tags:
  [INPUT]     - Input operands
  [PARTIAL_i] - i-th partial product (shifted copy of x when y's bit i is 1)
  [ACC]       - Running accumulator after adding each partial product
  [OUTPUT]    - Final product
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import IntEnum


class SchoolStepType(IntEnum):
    """Step types in a school multiplication trace."""
    INPUT = 0
    PARTIAL = 1
    ACC = 2
    OUTPUT = 3


SCHOOL_STEP_TAG = {
    SchoolStepType.INPUT: "[INPUT]",
    SchoolStepType.PARTIAL: "[PARTIAL]",
    SchoolStepType.ACC: "[ACC]",
    SchoolStepType.OUTPUT: "[OUTPUT]",
}


@dataclass
class SchoolTraceStep:
    """A single step in the school multiplication trace.

    Attributes:
        tag: The step type tag string.
        bits: List of bit values (0/1), LSB first.
        step_type: The SchoolStepType enum value.
        partial_index: For PARTIAL steps, which bit of y this corresponds to.
        bit_significance_offset: Bit offset for position encoding.
        description: Human-readable description.
    """
    tag: str
    bits: List[int]
    step_type: SchoolStepType
    partial_index: int = -1
    bit_significance_offset: int = 0
    description: str = ""


@dataclass
class SchoolTrace:
    """A complete school multiplication trace.

    Attributes:
        x: First operand.
        y: Second operand.
        x_bits: Number of bits for x.
        y_bits: Number of bits for y.
        expected_product: x * y.
        steps: Ordered list of trace steps.
        trace_product: The product computed by the trace.
    """
    x: int
    y: int
    x_bits: int
    y_bits: int
    expected_product: int
    steps: List[SchoolTraceStep] = field(default_factory=list)
    trace_product: Optional[int] = None

    def verify(self) -> bool:
        """Verify trace produces correct result."""
        return self.trace_product == self.expected_product


def int_to_bits(value: int, n_bits: int) -> List[int]:
    """Convert a non-negative integer to a list of bits (LSB first)."""
    assert value >= 0, f"Value must be non-negative, got {value}"
    bits = []
    for i in range(n_bits):
        bits.append((value >> i) & 1)
    return bits


def bits_to_int(bits: List[int]) -> int:
    """Convert a list of bits (LSB first) to a non-negative integer."""
    value = 0
    for i, b in enumerate(bits):
        value += b * (1 << i)
    return value


class SchoolTraceGenerator:
    """Generates school (long) multiplication traces in binary representation.

    The trace shows:
    1. Input operands
    2. Each partial product (x shifted left by i positions when y's bit i is 1)
    3. Running accumulator after adding each partial product
    4. Final output

    This is the O(n^2) baseline algorithm.
    """

    def __init__(self):
        pass

    def generate(self, x: int, y: int, n_bits: int) -> SchoolTrace:
        """Generate a school multiplication trace for x * y.

        Args:
            x: First operand (non-negative).
            y: Second operand (non-negative).
            n_bits: Bit width for operands.

        Returns:
            SchoolTrace with all steps and verified result.
        """
        assert x >= 0 and y >= 0, "Operands must be non-negative"
        assert x < (1 << n_bits), f"x={x} doesn't fit in {n_bits} bits"
        assert y < (1 << n_bits), f"y={y} doesn't fit in {n_bits} bits"

        product_bits = 2 * n_bits

        trace = SchoolTrace(
            x=x, y=y, x_bits=n_bits, y_bits=n_bits,
            expected_product=x * y
        )

        # INPUT step
        x_bits_list = int_to_bits(x, n_bits)
        y_bits_list = int_to_bits(y, n_bits)

        trace.steps.append(SchoolTraceStep(
            tag="[INPUT]",
            bits=x_bits_list + y_bits_list,
            step_type=SchoolStepType.INPUT,
            description=f"Input: {x} * {y} ({n_bits}-bit)"
        ))

        # Generate partial products and running accumulator
        accumulator = 0

        for i in range(n_bits):
            y_bit = (y >> i) & 1

            if y_bit == 1:
                # Partial product: x shifted left by i
                partial = x << i
            else:
                # Partial product is 0 (we still show it in the trace)
                partial = 0

            partial_bits = int_to_bits(partial, product_bits)

            trace.steps.append(SchoolTraceStep(
                tag=f"[PARTIAL]",
                bits=partial_bits,
                step_type=SchoolStepType.PARTIAL,
                partial_index=i,
                bit_significance_offset=i,
                description=(
                    f"Partial product {i}: y[{i}]={y_bit}, "
                    f"partial = {partial} (x << {i} if y[{i}]=1)"
                )
            ))

            # Update accumulator
            accumulator += partial

            acc_bits = int_to_bits(accumulator, product_bits)
            trace.steps.append(SchoolTraceStep(
                tag="[ACC]",
                bits=acc_bits,
                step_type=SchoolStepType.ACC,
                partial_index=i,
                description=f"Accumulator after bit {i}: {accumulator}"
            ))

        # OUTPUT step
        product_bits_list = int_to_bits(accumulator, product_bits)
        trace.steps.append(SchoolTraceStep(
            tag="[OUTPUT]",
            bits=product_bits_list,
            step_type=SchoolStepType.OUTPUT,
            description=f"Output: {accumulator}"
        ))

        trace.trace_product = accumulator
        assert trace.verify(), (
            f"School trace verification failed: {accumulator} != {x * y}"
        )

        return trace

    def trace_to_string(
        self, trace: SchoolTrace, show_description: bool = False
    ) -> str:
        """Convert a trace to a human-readable string.

        Args:
            trace: The SchoolTrace to format.
            show_description: If True, add descriptions as comments.

        Returns:
            Multi-line string representation.
        """
        lines = []
        for step in trace.steps:
            bits_str = "".join(str(b) for b in reversed(step.bits))
            tag = step.tag
            if step.step_type == SchoolStepType.PARTIAL:
                tag = f"[PARTIAL_{step.partial_index}]"
            line = f"{tag} {bits_str}"
            if show_description and step.description:
                line += f"  # {step.description}"
            lines.append(line)
        return "\n".join(lines)

    def trace_to_token_sequence(self, trace: SchoolTrace) -> List:
        """Convert a trace to a flat token sequence with position metadata.

        Returns a list of (token, position_metadata) tuples where:
        - token: either a tag string or a bit value (0 or 1)
        - position_metadata: dict with keys:
            'bit_significance': int
            'recursion_depth': 0 (always, since school has no recursion)
            'sub_problem_id': () (always, since school has no sub-problems)
            'step_type': SchoolStepType
        """
        sequence = []
        for step in trace.steps:
            # Tag token
            tag = step.tag
            if step.step_type == SchoolStepType.PARTIAL:
                tag = f"[PARTIAL_{step.partial_index}]"

            sequence.append((
                tag,
                {
                    'bit_significance': -1,
                    'recursion_depth': 0,
                    'sub_problem_id': (),
                    'step_type': step.step_type,
                }
            ))
            # Bit tokens
            for i, bit in enumerate(step.bits):
                sequence.append((
                    bit,
                    {
                        'bit_significance': i,
                        'recursion_depth': 0,
                        'sub_problem_id': (),
                        'step_type': step.step_type,
                    }
                ))
        return sequence


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_school_trace():
    """Run comprehensive tests for school multiplication traces."""
    import random

    print("Testing school multiplication trace generator...")

    gen = SchoolTraceGenerator()

    # Exhaustive test for 4-bit * 4-bit
    n_tested = 0
    for a in range(16):
        for b in range(16):
            trace = gen.generate(a, b, 4)
            assert trace.verify(), f"FAIL: {a} * {b} = {a*b}, got {trace.trace_product}"
            n_tested += 1
    print(f"  4-bit exhaustive: {n_tested} tests PASSED")

    # Test 8-bit random
    random.seed(42)
    for _ in range(500):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        trace = gen.generate(a, b, 8)
        assert trace.verify(), f"FAIL: {a} * {b} = {a*b}, got {trace.trace_product}"
    print("  8-bit: 500 random pairs PASSED")

    # Test 16-bit random
    random.seed(42)
    for _ in range(100):
        a = random.randint(0, 65535)
        b = random.randint(0, 65535)
        trace = gen.generate(a, b, 16)
        assert trace.verify(), f"FAIL: {a} * {b} = {a*b}, got {trace.trace_product}"
    print("  16-bit: 100 random pairs PASSED")

    # Test edge cases
    for a, b in [(0, 0), (0, 255), (255, 0), (1, 1), (255, 255)]:
        trace = gen.generate(a, b, 8)
        assert trace.verify(), f"Edge case FAIL: {a} * {b}"
    print("  Edge cases: PASSED")

    # Test trace string
    trace = gen.generate(7, 5, 4)
    s = gen.trace_to_string(trace, show_description=True)
    assert "[INPUT]" in s
    assert "[PARTIAL_" in s
    assert "[ACC]" in s
    assert "[OUTPUT]" in s
    print("  Trace string representation: PASSED")

    # Test token sequence
    trace = gen.generate(7, 5, 4)
    seq = gen.trace_to_token_sequence(trace)
    assert len(seq) > 0
    for token, meta in seq:
        assert isinstance(token, (int, str))
        if isinstance(token, int):
            assert token in (0, 1)
    print("  Token sequence generation: PASSED")

    # Verify trace length scales as expected (O(n) partial products)
    trace_4 = gen.generate(15, 15, 4)
    trace_8 = gen.generate(255, 255, 8)
    # 8-bit should have roughly twice as many PARTIAL steps
    partials_4 = sum(1 for s in trace_4.steps if s.step_type == SchoolStepType.PARTIAL)
    partials_8 = sum(1 for s in trace_8.steps if s.step_type == SchoolStepType.PARTIAL)
    assert partials_4 == 4, f"Expected 4 partial products for 4-bit, got {partials_4}"
    assert partials_8 == 8, f"Expected 8 partial products for 8-bit, got {partials_8}"
    print("  Trace length scaling: PASSED")

    print("\nAll school trace tests PASSED!")


if __name__ == "__main__":
    _test_school_trace()
