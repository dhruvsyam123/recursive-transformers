"""
Karatsuba recursion trace generator.

Generates full structured traces of the Karatsuba multiplication algorithm
in binary representation, with configurable base case size.

Each trace is a list of TraceStep objects with tags:
  [INPUT], [SPLIT], [SUB_MUL_0], [SUB_MUL_1], [SUB_MUL_2],
  [ADD], [SUB], [COMBINE], [OUTPUT], [BASE_MUL]

Supports both depth-first and breadth-first trace orderings.
Includes position metadata: (bit_significance, recursion_depth, sub_problem_id, step_type)

IMPORTANT DESIGN NOTE on the z1 sub-problem:
  Karatsuba computes z1_raw = (x_lo + x_hi) * (y_lo + y_hi).
  When x and y are n-bit numbers split into half = n/2-bit halves,
  the sums (x_lo + x_hi) and (y_lo + y_hi) can be (half+1) bits.
  For the z1 sub-multiplication, we use (half+1) bits directly.
  The recursion handles general (not just power-of-2) bit widths:
  split uses half = n_bits // 2 for the low part and (n_bits - half)
  for the high part. This guarantees termination since both halves
  are strictly smaller than n_bits for n_bits >= 2.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import IntEnum


class StepType(IntEnum):
    """Enumeration of step types in a Karatsuba trace."""
    INPUT = 0
    SPLIT = 1
    SUB_MUL_0 = 2
    SUB_MUL_1 = 3
    SUB_MUL_2 = 4
    ADD = 5
    SUB = 6
    COMBINE = 7
    OUTPUT = 8
    BASE_MUL = 9


# Map from StepType to string tag
STEP_TAG = {
    StepType.INPUT: "[INPUT]",
    StepType.SPLIT: "[SPLIT]",
    StepType.SUB_MUL_0: "[SUB_MUL_0]",
    StepType.SUB_MUL_1: "[SUB_MUL_1]",
    StepType.SUB_MUL_2: "[SUB_MUL_2]",
    StepType.ADD: "[ADD]",
    StepType.SUB: "[SUB]",
    StepType.COMBINE: "[COMBINE]",
    StepType.OUTPUT: "[OUTPUT]",
    StepType.BASE_MUL: "[BASE_MUL]",
}


@dataclass
class TraceStep:
    """A single step in the Karatsuba recursion trace.

    Attributes:
        tag: The step type tag string, e.g. "[SPLIT]".
        bits: List of bit values (0/1) representing the data for this step.
              For binary numbers, bits[0] is LSB, bits[-1] is MSB.
        step_type: The StepType enum value.
        recursion_depth: How deep in the recursion tree (0 = top level).
        sub_problem_id: Path through the recursion tree, e.g. (0,) for z0,
                        (1, 2) for z2 within z1's sub-problem.
        bit_significance_offset: The bit offset of this sub-problem's LSB
                                  within the top-level number.
        description: Human-readable description of what this step does.
    """
    tag: str
    bits: List[int]
    step_type: StepType
    recursion_depth: int = 0
    sub_problem_id: Tuple[int, ...] = ()
    bit_significance_offset: int = 0
    description: str = ""


@dataclass
class KaratsubaTrace:
    """A complete Karatsuba multiplication trace.

    Attributes:
        x: First operand (integer).
        y: Second operand (integer).
        x_bits: Number of bits used for x.
        y_bits: Number of bits used for y.
        expected_product: x * y.
        steps: Ordered list of TraceStep objects.
        trace_product: The product computed by the trace (should == expected_product).
    """
    x: int
    y: int
    x_bits: int
    y_bits: int
    expected_product: int
    steps: List[TraceStep] = field(default_factory=list)
    trace_product: Optional[int] = None

    def verify(self) -> bool:
        """Verify the trace produces the correct multiplication result."""
        return self.trace_product == self.expected_product


def int_to_bits(value: int, n_bits: int) -> List[int]:
    """Convert a non-negative integer to a list of bits (LSB first).

    Args:
        value: Non-negative integer to convert.
        n_bits: Number of bits to use (zero-padded).

    Returns:
        List of 0s and 1s, with bits[0] = LSB.
    """
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


def required_bits(value: int) -> int:
    """Return the minimum number of bits needed to represent a non-negative integer."""
    if value == 0:
        return 1
    return value.bit_length()


class KaratsubaTraceGenerator:
    """Generates Karatsuba multiplication traces in binary representation.

    Args:
        base_case_bits: Size (in bits) at or below which multiplication is
                        done directly (no further recursion). Must be >= 1.
                        Typical values: 1, 2, 4.
    """

    def __init__(self, base_case_bits: int = 4):
        assert base_case_bits >= 1, "Base case bits must be >= 1"
        self.base_case_bits = base_case_bits

    def generate(self, x: int, y: int, n_bits: int,
                 ordering: str = "depth_first") -> KaratsubaTrace:
        """Generate a complete Karatsuba trace for x * y.

        Args:
            x: First operand (non-negative integer).
            y: Second operand (non-negative integer).
            n_bits: Bit width to use for operands (must be a power of 2 and
                    >= base_case_bits). Both x and y are zero-padded to n_bits.
            ordering: "depth_first" or "breadth_first".

        Returns:
            KaratsubaTrace with all steps and verified result.
        """
        assert x >= 0 and y >= 0, "Operands must be non-negative"
        assert n_bits >= self.base_case_bits, (
            f"n_bits ({n_bits}) must be >= base_case_bits ({self.base_case_bits})"
        )
        assert x < (1 << n_bits), f"x={x} doesn't fit in {n_bits} bits"
        assert y < (1 << n_bits), f"y={y} doesn't fit in {n_bits} bits"
        # Top-level n_bits should be a power of 2 for clean recursive splitting
        assert n_bits & (n_bits - 1) == 0, (
            f"n_bits ({n_bits}) should be a power of 2 for clean splitting"
        )

        trace = KaratsubaTrace(
            x=x, y=y, x_bits=n_bits, y_bits=n_bits,
            expected_product=x * y
        )

        if ordering == "depth_first":
            result = self._generate_depth_first(
                x, y, n_bits, trace, depth=0, sub_id=(), bit_offset=0
            )
        elif ordering == "breadth_first":
            result = self._generate_breadth_first(x, y, n_bits, trace)
        else:
            raise ValueError(f"Unknown ordering: {ordering}")

        trace.trace_product = result
        assert trace.verify(), (
            f"Trace verification failed: trace_product={trace.trace_product} "
            f"!= expected={trace.expected_product} for {x} * {y}"
        )
        return trace

    def _generate_depth_first(
        self, x: int, y: int, n_bits: int,
        trace: KaratsubaTrace, depth: int, sub_id: Tuple[int, ...],
        bit_offset: int
    ) -> int:
        """Generate a depth-first Karatsuba trace recursively.

        Handles general n_bits (not just powers of 2) for internal recursion.
        The split uses half = n_bits // 2 for the low part.

        Returns the computed product (for verification).
        """
        # Product bit width: multiplying two n-bit numbers gives up to 2*n bits
        product_bits = 2 * n_bits

        x_bits_list = int_to_bits(x, n_bits)
        y_bits_list = int_to_bits(y, n_bits)

        # INPUT step
        trace.steps.append(TraceStep(
            tag="[INPUT]",
            bits=x_bits_list + y_bits_list,
            step_type=StepType.INPUT,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=bit_offset,
            description=f"Input: {x} * {y} ({n_bits}-bit)"
        ))

        # Base case: multiply directly
        if n_bits <= self.base_case_bits:
            product = x * y
            product_bits_list = int_to_bits(product, product_bits)
            trace.steps.append(TraceStep(
                tag="[BASE_MUL]",
                bits=product_bits_list,
                step_type=StepType.BASE_MUL,
                recursion_depth=depth,
                sub_problem_id=sub_id,
                bit_significance_offset=bit_offset,
                description=f"Base case: {x} * {y} = {product}"
            ))
            trace.steps.append(TraceStep(
                tag="[OUTPUT]",
                bits=product_bits_list,
                step_type=StepType.OUTPUT,
                recursion_depth=depth,
                sub_problem_id=sub_id,
                bit_significance_offset=bit_offset,
                description=f"Output: {product} ({product_bits}-bit)"
            ))
            return product

        # Recursive case: Karatsuba decomposition
        # half = number of low bits; hi gets (n_bits - half) bits
        half = n_bits // 2
        hi_bits = n_bits - half  # == half when n_bits is even, half+1 when odd
        mask_low = (1 << half) - 1

        # Split
        x_lo = x & mask_low
        x_hi = x >> half
        y_lo = y & mask_low
        y_hi = y >> half

        # x_lo, y_lo are half-bit; x_hi, y_hi are hi_bits-bit
        x_lo_bits = int_to_bits(x_lo, half)
        x_hi_bits = int_to_bits(x_hi, hi_bits)
        y_lo_bits = int_to_bits(y_lo, half)
        y_hi_bits = int_to_bits(y_hi, hi_bits)

        trace.steps.append(TraceStep(
            tag="[SPLIT]",
            bits=x_hi_bits + x_lo_bits + y_hi_bits + y_lo_bits,
            step_type=StepType.SPLIT,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=bit_offset,
            description=(
                f"Split: x_hi={x_hi}({hi_bits}b), x_lo={x_lo}({half}b), "
                f"y_hi={y_hi}({hi_bits}b), y_lo={y_lo}({half}b)"
            )
        ))

        # For z0 and z2, the sub-problem sizes are the larger of the two halves
        # to ensure both operands fit. z0 = x_lo * y_lo uses half bits.
        # z2 = x_hi * y_hi uses hi_bits bits.
        # For z1, the sums (x_lo + x_hi) and (y_lo + y_hi) need at most
        # max(half, hi_bits) + 1 bits.

        # Determine sub-problem sizes for z0 and z2.
        # z0 operands are half-bit numbers.
        z0_n_bits = self._sub_problem_size(half)
        # z2 operands are hi_bits-bit numbers.
        z2_n_bits = self._sub_problem_size(hi_bits)

        # Sub-multiplication 0: z0 = x_lo * y_lo
        trace.steps.append(TraceStep(
            tag="[SUB_MUL_0]",
            bits=int_to_bits(x_lo, z0_n_bits) + int_to_bits(y_lo, z0_n_bits),
            step_type=StepType.SUB_MUL_0,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=bit_offset,
            description=f"Sub-mul 0: z0 = {x_lo} * {y_lo} ({z0_n_bits}-bit)"
        ))
        z0 = self._generate_depth_first(
            x_lo, y_lo, z0_n_bits, trace, depth + 1, sub_id + (0,),
            bit_offset
        )

        # Sub-multiplication 2: z2 = x_hi * y_hi
        trace.steps.append(TraceStep(
            tag="[SUB_MUL_2]",
            bits=int_to_bits(x_hi, z2_n_bits) + int_to_bits(y_hi, z2_n_bits),
            step_type=StepType.SUB_MUL_2,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=bit_offset + half,
            description=f"Sub-mul 2: z2 = {x_hi} * {y_hi} ({z2_n_bits}-bit)"
        ))
        z2 = self._generate_depth_first(
            x_hi, y_hi, z2_n_bits, trace, depth + 1, sub_id + (2,),
            bit_offset + half
        )

        # ADD step: compute (x_lo + x_hi) and (y_lo + y_hi)
        sum_x = x_lo + x_hi
        sum_y = y_lo + y_hi
        # The sums need at most max(half, hi_bits) + 1 bits
        sum_actual_bits = max(half, hi_bits) + 1
        sum_display_bits = sum_actual_bits  # bits shown in the ADD step

        sum_x_bits = int_to_bits(sum_x, sum_display_bits)
        sum_y_bits = int_to_bits(sum_y, sum_display_bits)

        trace.steps.append(TraceStep(
            tag="[ADD]",
            bits=sum_x_bits + sum_y_bits,
            step_type=StepType.ADD,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=bit_offset,
            description=(
                f"Add: (x_lo+x_hi)={sum_x}, (y_lo+y_hi)={sum_y} ({sum_display_bits}-bit)"
            )
        ))

        # Sub-multiplication 1: z1_raw = (x_lo + x_hi) * (y_lo + y_hi)
        z1_n_bits = self._sub_problem_size(sum_actual_bits)

        # Guard against infinite recursion: when z1_n_bits >= n_bits
        # (happens for n_bits=2 or 3), the sub-problem doesn't shrink.
        # Force it to be a base case in that situation.
        z1_force_base = z1_n_bits >= n_bits

        trace.steps.append(TraceStep(
            tag="[SUB_MUL_1]",
            bits=int_to_bits(sum_x, z1_n_bits) + int_to_bits(sum_y, z1_n_bits),
            step_type=StepType.SUB_MUL_1,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=bit_offset,
            description=f"Sub-mul 1: z1_raw = {sum_x} * {sum_y} ({z1_n_bits}-bit)"
        ))

        if z1_force_base:
            # Compute directly to avoid infinite recursion
            z1_raw = sum_x * sum_y
            z1_raw_bits = int_to_bits(z1_raw, 2 * z1_n_bits)
            trace.steps.append(TraceStep(
                tag="[BASE_MUL]",
                bits=z1_raw_bits,
                step_type=StepType.BASE_MUL,
                recursion_depth=depth + 1,
                sub_problem_id=sub_id + (1,),
                bit_significance_offset=bit_offset,
                description=f"Base case (forced): {sum_x} * {sum_y} = {z1_raw}"
            ))
            trace.steps.append(TraceStep(
                tag="[OUTPUT]",
                bits=z1_raw_bits,
                step_type=StepType.OUTPUT,
                recursion_depth=depth + 1,
                sub_problem_id=sub_id + (1,),
                bit_significance_offset=bit_offset,
                description=f"Output: {z1_raw}"
            ))
        else:
            z1_raw = self._generate_depth_first(
                sum_x, sum_y, z1_n_bits, trace, depth + 1, sub_id + (1,),
                bit_offset
            )

        # SUB step: z1 = z1_raw - z0 - z2
        z1 = z1_raw - z0 - z2
        assert z1 >= 0, (
            f"z1 should be non-negative: z1_raw={z1_raw}, z0={z0}, z2={z2}"
        )

        z0_result_bits = 2 * z0_n_bits
        z2_result_bits = 2 * z2_n_bits
        z1_raw_result_bits = 2 * z1_n_bits

        trace.steps.append(TraceStep(
            tag="[SUB]",
            bits=(
                int_to_bits(z1_raw, z1_raw_result_bits)
                + int_to_bits(z0, z0_result_bits)
                + int_to_bits(z2, z2_result_bits)
                + int_to_bits(z1, product_bits)
            ),
            step_type=StepType.SUB,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=bit_offset,
            description=f"Sub: z1 = {z1_raw} - {z0} - {z2} = {z1}"
        ))

        # COMBINE: result = z2 * 2^n_bits + z1 * 2^half + z0
        # Note: we use n_bits for the z2 shift (not 2*half) because
        # X = x_hi * 2^half + x_lo, so X*Y = z2 * 2^(2*half) + z1 * 2^half + z0
        # But half might not be n_bits/2 exactly for odd n_bits.
        # The shift for z2 is always 2*half (the combined width of both low parts).
        product = z0 + (z1 << half) + (z2 << (2 * half))
        product_bits_list = int_to_bits(product, product_bits)

        trace.steps.append(TraceStep(
            tag="[COMBINE]",
            bits=product_bits_list,
            step_type=StepType.COMBINE,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=bit_offset,
            description=(
                f"Combine: {z2}*2^{2*half} + {z1}*2^{half} + {z0} = {product}"
            )
        ))

        # OUTPUT
        trace.steps.append(TraceStep(
            tag="[OUTPUT]",
            bits=product_bits_list,
            step_type=StepType.OUTPUT,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=bit_offset,
            description=f"Output: {product} ({product_bits}-bit)"
        ))

        return product

    def _generate_breadth_first(
        self, x: int, y: int, n_bits: int,
        trace: KaratsubaTrace
    ) -> int:
        """Generate a breadth-first Karatsuba trace.

        Processes all sub-problems at the same recursion level before going deeper.
        The trace has a "decompose" phase (top-down) and a "combine" phase
        (bottom-up).

        Returns the computed product (for verification).
        """
        levels = []
        results = {}

        # BFS decomposition
        queue = [(x, y, n_bits, (), 0)]
        depth = 0

        while queue:
            current_level = queue
            queue = []
            level_steps = []

            for (xi, yi, ni, sid, boff) in current_level:
                xi_bits = int_to_bits(xi, ni)
                yi_bits = int_to_bits(yi, ni)

                level_steps.append(TraceStep(
                    tag="[INPUT]",
                    bits=xi_bits + yi_bits,
                    step_type=StepType.INPUT,
                    recursion_depth=depth,
                    sub_problem_id=sid,
                    bit_significance_offset=boff,
                    description=f"Input: {xi} * {yi} ({ni}-bit)"
                ))

                if ni <= self.base_case_bits:
                    product = xi * yi
                    product_bits_list = int_to_bits(product, 2 * ni)
                    level_steps.append(TraceStep(
                        tag="[BASE_MUL]",
                        bits=product_bits_list,
                        step_type=StepType.BASE_MUL,
                        recursion_depth=depth,
                        sub_problem_id=sid,
                        bit_significance_offset=boff,
                        description=f"Base case: {xi} * {yi} = {product}"
                    ))
                    level_steps.append(TraceStep(
                        tag="[OUTPUT]",
                        bits=product_bits_list,
                        step_type=StepType.OUTPUT,
                        recursion_depth=depth,
                        sub_problem_id=sid,
                        bit_significance_offset=boff,
                        description=f"Output: {product}"
                    ))
                    results[sid] = {
                        'product': product,
                        'n_bits': ni,
                        'is_base': True,
                    }
                else:
                    half = ni // 2
                    hi_bits = ni - half
                    mask_low = (1 << half) - 1
                    xi_lo = xi & mask_low
                    xi_hi = xi >> half
                    yi_lo = yi & mask_low
                    yi_hi = yi >> half

                    level_steps.append(TraceStep(
                        tag="[SPLIT]",
                        bits=(
                            int_to_bits(xi_hi, hi_bits) + int_to_bits(xi_lo, half)
                            + int_to_bits(yi_hi, hi_bits) + int_to_bits(yi_lo, half)
                        ),
                        step_type=StepType.SPLIT,
                        recursion_depth=depth,
                        sub_problem_id=sid,
                        bit_significance_offset=boff,
                        description=(
                            f"Split: x_hi={xi_hi}, x_lo={xi_lo}, "
                            f"y_hi={yi_hi}, y_lo={yi_lo}"
                        )
                    ))

                    sum_x = xi_lo + xi_hi
                    sum_y = yi_lo + yi_hi
                    sum_actual_bits = max(half, hi_bits) + 1
                    sum_display_bits = sum_actual_bits

                    level_steps.append(TraceStep(
                        tag="[ADD]",
                        bits=(int_to_bits(sum_x, sum_display_bits)
                              + int_to_bits(sum_y, sum_display_bits)),
                        step_type=StepType.ADD,
                        recursion_depth=depth,
                        sub_problem_id=sid,
                        bit_significance_offset=boff,
                        description=f"Add: (x_lo+x_hi)={sum_x}, (y_lo+y_hi)={sum_y}"
                    ))

                    z0_n_bits = self._sub_problem_size(half)
                    z2_n_bits = self._sub_problem_size(hi_bits)
                    z1_n_bits = self._sub_problem_size(sum_actual_bits)

                    # z0
                    level_steps.append(TraceStep(
                        tag="[SUB_MUL_0]",
                        bits=int_to_bits(xi_lo, z0_n_bits) + int_to_bits(yi_lo, z0_n_bits),
                        step_type=StepType.SUB_MUL_0,
                        recursion_depth=depth,
                        sub_problem_id=sid,
                        bit_significance_offset=boff,
                        description=f"Sub-mul 0: z0 = {xi_lo} * {yi_lo}"
                    ))
                    queue.append((xi_lo, yi_lo, z0_n_bits, sid + (0,), boff))

                    # z2
                    level_steps.append(TraceStep(
                        tag="[SUB_MUL_2]",
                        bits=int_to_bits(xi_hi, z2_n_bits) + int_to_bits(yi_hi, z2_n_bits),
                        step_type=StepType.SUB_MUL_2,
                        recursion_depth=depth,
                        sub_problem_id=sid,
                        bit_significance_offset=boff + half,
                        description=f"Sub-mul 2: z2 = {xi_hi} * {yi_hi}"
                    ))
                    queue.append((xi_hi, yi_hi, z2_n_bits, sid + (2,), boff + half))

                    # z1 — guard against z1_n_bits >= ni (infinite recursion)
                    z1_force_base = z1_n_bits >= ni
                    level_steps.append(TraceStep(
                        tag="[SUB_MUL_1]",
                        bits=(int_to_bits(sum_x, z1_n_bits)
                              + int_to_bits(sum_y, z1_n_bits)),
                        step_type=StepType.SUB_MUL_1,
                        recursion_depth=depth,
                        sub_problem_id=sid,
                        bit_significance_offset=boff,
                        description=f"Sub-mul 1: z1_raw = {sum_x} * {sum_y}"
                    ))

                    if z1_force_base:
                        # Compute directly
                        z1_product = sum_x * sum_y
                        z1_prod_bits = int_to_bits(z1_product, 2 * z1_n_bits)
                        level_steps.append(TraceStep(
                            tag="[BASE_MUL]",
                            bits=z1_prod_bits,
                            step_type=StepType.BASE_MUL,
                            recursion_depth=depth + 1,
                            sub_problem_id=sid + (1,),
                            bit_significance_offset=boff,
                            description=f"Base case (forced): {sum_x} * {sum_y} = {z1_product}"
                        ))
                        level_steps.append(TraceStep(
                            tag="[OUTPUT]",
                            bits=z1_prod_bits,
                            step_type=StepType.OUTPUT,
                            recursion_depth=depth + 1,
                            sub_problem_id=sid + (1,),
                            bit_significance_offset=boff,
                            description=f"Output: {z1_product}"
                        ))
                        results[sid + (1,)] = {
                            'product': z1_product,
                            'n_bits': z1_n_bits,
                            'is_base': True,
                        }
                    else:
                        queue.append((sum_x, sum_y, z1_n_bits, sid + (1,), boff))

                    results[sid] = {
                        'n_bits': ni,
                        'half': half,
                        'z0_n_bits': z0_n_bits,
                        'z2_n_bits': z2_n_bits,
                        'z1_n_bits': z1_n_bits,
                        'is_base': False,
                        'sub_ids': (sid + (0,), sid + (1,), sid + (2,)),
                    }

            levels.append(level_steps)
            depth += 1

        # Add all decomposition steps (top-down) to trace
        for level_steps in levels:
            trace.steps.extend(level_steps)

        # Combine bottom-up
        self._combine_bottom_up(results, (), trace)

        return results[()]['product']

    def _combine_bottom_up(
        self, results: dict, sub_id: tuple, trace: KaratsubaTrace
    ):
        """Recursively combine results bottom-up for breadth-first trace."""
        info = results[sub_id]
        if info['is_base']:
            return

        for child_sid in info['sub_ids']:
            self._combine_bottom_up(results, child_sid, trace)

        z0 = results[info['sub_ids'][0]]['product']
        z1_raw = results[info['sub_ids'][1]]['product']
        z2 = results[info['sub_ids'][2]]['product']

        half = info['half']
        ni = info['n_bits']
        product_bits = 2 * ni
        z0_n_bits = info['z0_n_bits']
        z2_n_bits = info['z2_n_bits']
        z1_n_bits = info['z1_n_bits']

        z1 = z1_raw - z0 - z2
        assert z1 >= 0, (
            f"z1 must be non-negative: z1_raw={z1_raw}, z0={z0}, z2={z2}"
        )

        depth = len(sub_id)

        trace.steps.append(TraceStep(
            tag="[SUB]",
            bits=(
                int_to_bits(z1_raw, 2 * z1_n_bits)
                + int_to_bits(z0, 2 * z0_n_bits)
                + int_to_bits(z2, 2 * z2_n_bits)
                + int_to_bits(z1, product_bits)
            ),
            step_type=StepType.SUB,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=0,
            description=f"Sub: z1 = {z1_raw} - {z0} - {z2} = {z1}"
        ))

        product = z0 + (z1 << half) + (z2 << (2 * half))

        trace.steps.append(TraceStep(
            tag="[COMBINE]",
            bits=int_to_bits(product, product_bits),
            step_type=StepType.COMBINE,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=0,
            description=(
                f"Combine: {z2}*2^{2*half} + {z1}*2^{half} + {z0} = {product}"
            )
        ))

        trace.steps.append(TraceStep(
            tag="[OUTPUT]",
            bits=int_to_bits(product, product_bits),
            step_type=StepType.OUTPUT,
            recursion_depth=depth,
            sub_problem_id=sub_id,
            bit_significance_offset=0,
            description=f"Output: {product}"
        ))

        info['product'] = product

    def _sub_problem_size(self, n: int) -> int:
        """Determine the bit width for a sub-problem with n-bit operands.

        If n <= base_case_bits, return n (it will be a base case).
        Otherwise return n as-is; the recursion handles general bit widths
        by splitting n_bits // 2 for the low part and n_bits - n_bits // 2
        for the high part, guaranteeing strict decrease.
        """
        return max(n, 1)

    def trace_to_string(self, trace: KaratsubaTrace, show_description: bool = False) -> str:
        """Convert a trace to a human-readable string representation.

        Args:
            trace: The KaratsubaTrace to format.
            show_description: If True, add descriptions as comments.

        Returns:
            Multi-line string representation of the trace.
        """
        lines = []
        for step in trace.steps:
            indent = "  " * step.recursion_depth
            bits_str = "".join(str(b) for b in reversed(step.bits))  # MSB first for display
            line = f"{indent}{step.tag} {bits_str}"
            if show_description and step.description:
                line += f"  # {step.description}"
            lines.append(line)
        return "\n".join(lines)

    def trace_to_token_sequence(self, trace: KaratsubaTrace) -> List:
        """Convert a trace to a flat token sequence with position metadata.

        Returns a list of (token, position_metadata) tuples where:
        - token: either a tag string or a bit value (0 or 1)
        - position_metadata: dict with keys:
            'bit_significance': int (bit position within the sub-problem)
            'recursion_depth': int
            'sub_problem_id': tuple
            'step_type': StepType
        """
        sequence = []
        for step in trace.steps:
            # Add the tag token
            sequence.append((
                step.tag,
                {
                    'bit_significance': -1,  # tag token has no bit significance
                    'recursion_depth': step.recursion_depth,
                    'sub_problem_id': step.sub_problem_id,
                    'step_type': step.step_type,
                }
            ))
            # Add bit tokens
            for i, bit in enumerate(step.bits):
                sequence.append((
                    bit,
                    {
                        'bit_significance': step.bit_significance_offset + i,
                        'recursion_depth': step.recursion_depth,
                        'sub_problem_id': step.sub_problem_id,
                        'step_type': step.step_type,
                    }
                ))
        return sequence


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_karatsuba_trace():
    """Run comprehensive tests to verify trace correctness."""
    import random

    print("Testing Karatsuba trace generator...")

    # Test with different base case sizes
    for base_bits in [1, 2, 4]:
        gen = KaratsubaTraceGenerator(base_case_bits=base_bits)

        # Test all 4-bit * 4-bit pairs (exhaustive)
        n_bits = max(4, base_bits)
        # Make sure n_bits is a power of 2
        if n_bits & (n_bits - 1) != 0:
            p = 1
            while p < n_bits:
                p <<= 1
            n_bits = p

        max_val = (1 << n_bits)
        test_pairs = []
        if n_bits <= 4:
            # Exhaustive for small sizes
            for a in range(max_val):
                for b in range(max_val):
                    test_pairs.append((a, b))
        else:
            # Sample for larger sizes
            for a in range(min(16, max_val)):
                for b in range(min(16, max_val)):
                    test_pairs.append((a, b))

        for ordering in ["depth_first", "breadth_first"]:
            n_tested = 0
            for a, b in test_pairs:
                trace = gen.generate(a, b, n_bits, ordering=ordering)
                assert trace.verify(), (
                    f"FAIL: {a} * {b} = {a*b} but trace got "
                    f"{trace.trace_product} (base={base_bits}, order={ordering})"
                )
                n_tested += 1

            print(
                f"  base_case={base_bits}, n_bits={n_bits}, "
                f"ordering={ordering}: {n_tested} tests PASSED"
            )

    # Test 8-bit with base case 4
    gen = KaratsubaTraceGenerator(base_case_bits=4)
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        for ordering in ["depth_first", "breadth_first"]:
            trace = gen.generate(a, b, 8, ordering=ordering)
            assert trace.verify(), (
                f"FAIL: {a} * {b} = {a*b} but trace got {trace.trace_product}"
            )

    print("  8-bit (base=4): 200 random pairs x 2 orderings PASSED")

    # Test 16-bit with base case 4
    gen = KaratsubaTraceGenerator(base_case_bits=4)
    random.seed(42)
    for _ in range(50):
        a = random.randint(0, 65535)
        b = random.randint(0, 65535)
        trace = gen.generate(a, b, 16, ordering="depth_first")
        assert trace.verify(), (
            f"FAIL: {a} * {b} = {a*b} but trace got {trace.trace_product}"
        )
    print("  16-bit (base=4): 50 random pairs PASSED")

    # Test 16-bit with base case 2
    gen = KaratsubaTraceGenerator(base_case_bits=2)
    random.seed(42)
    for _ in range(50):
        a = random.randint(0, 65535)
        b = random.randint(0, 65535)
        trace = gen.generate(a, b, 16, ordering="depth_first")
        assert trace.verify(), (
            f"FAIL: {a} * {b} = {a*b} but trace got {trace.trace_product}"
        )
    print("  16-bit (base=2): 50 random pairs PASSED")

    # Test 32-bit with base case 4
    gen = KaratsubaTraceGenerator(base_case_bits=4)
    random.seed(42)
    for _ in range(20):
        a = random.randint(0, (1 << 32) - 1)
        b = random.randint(0, (1 << 32) - 1)
        trace = gen.generate(a, b, 32, ordering="depth_first")
        assert trace.verify(), (
            f"FAIL: {a} * {b} = {a*b} but trace got {trace.trace_product}"
        )
    print("  32-bit (base=4): 20 random pairs PASSED")

    # Test token sequence generation
    gen = KaratsubaTraceGenerator(base_case_bits=4)
    trace = gen.generate(13, 11, 4, ordering="depth_first")
    seq = gen.trace_to_token_sequence(trace)
    assert len(seq) > 0, "Token sequence should not be empty"
    # Verify all tokens are valid
    for token, meta in seq:
        assert isinstance(token, (int, str)), f"Invalid token type: {type(token)}"
        if isinstance(token, int):
            assert token in (0, 1), f"Bit token must be 0 or 1, got {token}"
        assert 'bit_significance' in meta
        assert 'recursion_depth' in meta
        assert 'sub_problem_id' in meta
        assert 'step_type' in meta
    print("  Token sequence generation: PASSED")

    # Test edge cases
    gen = KaratsubaTraceGenerator(base_case_bits=4)
    for a, b in [(0, 0), (0, 15), (15, 0), (1, 1), (15, 15)]:
        trace = gen.generate(a, b, 4, ordering="depth_first")
        assert trace.verify(), f"Edge case FAIL: {a} * {b}"
    print("  Edge cases (0*0, 0*15, 15*0, 1*1, 15*15): PASSED")

    # Test trace string representation
    gen = KaratsubaTraceGenerator(base_case_bits=4)
    trace = gen.generate(7, 5, 8, ordering="depth_first")
    s = gen.trace_to_string(trace, show_description=True)
    assert "[INPUT]" in s
    assert "[SPLIT]" in s
    assert "[COMBINE]" in s
    assert "[OUTPUT]" in s
    print("  Trace string representation: PASSED")

    print("\nAll Karatsuba trace tests PASSED!")


if __name__ == "__main__":
    _test_karatsuba_trace()
