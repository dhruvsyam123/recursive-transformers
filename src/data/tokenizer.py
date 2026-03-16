"""
Tokenizer and hierarchical position encoding for Karatsuba multiplication traces.

Token vocabulary:
  0, 1                     - Binary digit tokens
  [INPUT]                  - Input operands marker
  [SPLIT]                  - Split step marker
  [SUB_MUL_0]              - Sub-multiplication 0 (z0 = x_lo * y_lo)
  [SUB_MUL_1]              - Sub-multiplication 1 (z1_raw = sum_x * sum_y)
  [SUB_MUL_2]              - Sub-multiplication 2 (z2 = x_hi * y_hi)
  [ADD]                    - Addition step
  [SUB]                    - Subtraction step
  [COMBINE]                - Combine step
  [OUTPUT]                 - Output marker
  [PAD]                    - Padding token
  [SEP]                    - Separator token
  [BASE_MUL]               - Base case multiplication

Hierarchical position encoding:
  (bit_significance, recursion_depth, sub_problem_id, step_type)

Position coupling: tokens with the same bit significance in the same
sub-problem share position IDs, enabling the model to learn structural
correspondences.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np


# ---- Vocabulary ----

# Special tokens and their IDs
SPECIAL_TOKENS = [
    "[PAD]",       # 0: padding
    "[SEP]",       # 1: separator
    "[INPUT]",     # 2
    "[SPLIT]",     # 3
    "[SUB_MUL_0]", # 4
    "[SUB_MUL_1]", # 5
    "[SUB_MUL_2]", # 6
    "[ADD]",       # 7
    "[SUB]",       # 8
    "[COMBINE]",   # 9
    "[OUTPUT]",    # 10
    "[BASE_MUL]",  # 11
]

# Binary digit tokens
BIT_TOKENS = ["0", "1"]  # IDs: 12, 13

# School-algorithm specific tokens (for baseline comparison)
# We reserve IDs for partial product tags up to 128-bit
SCHOOL_PARTIAL_TOKENS = [f"[PARTIAL_{i}]" for i in range(128)]
SCHOOL_ACC_TOKEN = "[ACC]"

# Full vocabulary
VOCAB = SPECIAL_TOKENS + BIT_TOKENS + [SCHOOL_ACC_TOKEN] + SCHOOL_PARTIAL_TOKENS

# Build token-to-ID and ID-to-token mappings
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: token for idx, token in enumerate(VOCAB)}

# Convenience constants
PAD_ID = TOKEN_TO_ID["[PAD]"]
SEP_ID = TOKEN_TO_ID["[SEP]"]
BIT_0_ID = TOKEN_TO_ID["0"]
BIT_1_ID = TOKEN_TO_ID["1"]

VOCAB_SIZE = len(VOCAB)


class Tokenizer:
    """Tokenizer for Karatsuba and school multiplication traces.

    Handles encoding of trace sequences to token IDs and position tuples,
    and decoding back to human-readable form.
    """

    def __init__(self, max_recursion_depth: int = 8, max_bit_significance: int = 256):
        """Initialize the tokenizer.

        Args:
            max_recursion_depth: Maximum recursion depth to support.
            max_bit_significance: Maximum bit significance to support.
        """
        self.max_recursion_depth = max_recursion_depth
        self.max_bit_significance = max_bit_significance
        self.token_to_id = dict(TOKEN_TO_ID)
        self.id_to_token = dict(ID_TO_TOKEN)
        self.vocab_size = VOCAB_SIZE
        self.pad_id = PAD_ID
        self.sep_id = SEP_ID

    def encode_token(self, token) -> int:
        """Encode a single token (str or int) to its token ID.

        Args:
            token: Either a string tag like "[INPUT]" or an int (0 or 1).

        Returns:
            Integer token ID.
        """
        if isinstance(token, int):
            assert token in (0, 1), f"Bit token must be 0 or 1, got {token}"
            return self.token_to_id[str(token)]
        elif isinstance(token, str):
            if token in self.token_to_id:
                return self.token_to_id[token]
            else:
                raise ValueError(f"Unknown token: {token}")
        else:
            raise TypeError(f"Token must be str or int, got {type(token)}")

    def decode_token(self, token_id: int) -> str:
        """Decode a token ID back to its string representation."""
        if token_id in self.id_to_token:
            return self.id_to_token[token_id]
        else:
            raise ValueError(f"Unknown token ID: {token_id}")

    def encode_trace_sequence(
        self, token_sequence: List[Tuple[Any, Dict]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a trace token sequence into arrays of token IDs and positions.

        Args:
            token_sequence: List of (token, position_metadata) tuples as
                           produced by KaratsubaTraceGenerator.trace_to_token_sequence()
                           or SchoolTraceGenerator.trace_to_token_sequence().

        Returns:
            Tuple of:
                token_ids: np.ndarray of shape (seq_len,) with token IDs.
                positions: np.ndarray of shape (seq_len, 4) with position tuples:
                    positions[:, 0] = bit_significance
                    positions[:, 1] = recursion_depth
                    positions[:, 2] = sub_problem_id_hash (hashed to integer)
                    positions[:, 3] = step_type
        """
        seq_len = len(token_sequence)
        token_ids = np.zeros(seq_len, dtype=np.int32)
        positions = np.zeros((seq_len, 4), dtype=np.int32)

        for i, (token, meta) in enumerate(token_sequence):
            token_ids[i] = self.encode_token(token)
            positions[i, 0] = meta.get('bit_significance', -1)
            positions[i, 1] = meta.get('recursion_depth', 0)
            positions[i, 2] = self._hash_sub_problem_id(
                meta.get('sub_problem_id', ())
            )
            positions[i, 3] = int(meta.get('step_type', 0))

        return token_ids, positions

    def decode_token_ids(self, token_ids: np.ndarray) -> List[str]:
        """Decode an array of token IDs to a list of string tokens."""
        return [self.decode_token(int(tid)) for tid in token_ids]

    def _hash_sub_problem_id(self, sub_id: Tuple[int, ...]) -> int:
        """Hash a sub-problem ID tuple to a unique integer.

        Sub-problem IDs are tuples like (), (0,), (0, 2), (1, 0, 1), etc.
        Each element is in {0, 1, 2} (for z0, z1, z2).

        We encode this as a base-3 number with a prefix to distinguish depths:
          () -> 0
          (0,) -> 1, (1,) -> 2, (2,) -> 3
          (0,0) -> 4, (0,1) -> 5, (0,2) -> 6, (1,0) -> 7, ...
        This is: sum_{i=0}^{d-1} 3^i + value_in_base3

        For our purposes, we use a simpler scheme: offset = (3^d - 1) / 2 + base3_value
        """
        if len(sub_id) == 0:
            return 0

        # Compute base-3 value
        base3_value = 0
        for i, elem in enumerate(sub_id):
            base3_value = base3_value * 3 + elem

        # Offset for this depth level: 1 + 3 + 9 + ... + 3^(d-1) = (3^d - 1) / 2
        d = len(sub_id)
        offset = (3**d - 1) // 2 + 1  # +1 because 0 is reserved for ()

        return offset + base3_value

    def compute_position_coupling_ids(
        self, positions: np.ndarray
    ) -> np.ndarray:
        """Compute position coupling IDs for attention.

        Tokens with the same bit significance in the same sub-problem at
        the same recursion level should share position IDs, enabling the
        model to attend between structurally related tokens.

        Args:
            positions: np.ndarray of shape (seq_len, 4) with position tuples.

        Returns:
            np.ndarray of shape (seq_len,) with coupled position IDs.
            Tokens sharing the same coupled position ID can attend to each other
            more easily (via position coupling).
        """
        seq_len = positions.shape[0]
        coupling_ids = np.zeros(seq_len, dtype=np.int32)

        for i in range(seq_len):
            bit_sig = positions[i, 0]
            depth = positions[i, 1]
            sub_id = positions[i, 2]

            if bit_sig < 0:
                # Tag tokens: assign unique position based on sequence position
                # (no coupling for tag tokens)
                coupling_ids[i] = -(i + 1)  # negative to distinguish
            else:
                # Bit tokens: couple by (bit_significance, depth, sub_problem)
                # This way, the same bit in the same sub-problem across different
                # steps (INPUT, SPLIT, COMBINE, OUTPUT) shares a position.
                coupling_ids[i] = (
                    bit_sig
                    + self.max_bit_significance * depth
                    + self.max_bit_significance * self.max_recursion_depth * sub_id
                )

        return coupling_ids

    def pad_sequence(
        self,
        token_ids: np.ndarray,
        positions: np.ndarray,
        max_len: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pad a sequence to max_len.

        Args:
            token_ids: Shape (seq_len,).
            positions: Shape (seq_len, 4).
            max_len: Target length.

        Returns:
            Tuple of:
                padded_token_ids: Shape (max_len,).
                padded_positions: Shape (max_len, 4).
                mask: Shape (max_len,), 1.0 for real tokens, 0.0 for padding.
        """
        seq_len = len(token_ids)
        assert seq_len <= max_len, (
            f"Sequence length {seq_len} exceeds max_len {max_len}"
        )

        padded_ids = np.full(max_len, self.pad_id, dtype=np.int32)
        padded_positions = np.zeros((max_len, 4), dtype=np.int32)
        mask = np.zeros(max_len, dtype=np.float32)

        padded_ids[:seq_len] = token_ids
        padded_positions[:seq_len] = positions
        mask[:seq_len] = 1.0

        return padded_ids, padded_positions, mask

    def create_input_output_pair(
        self,
        token_ids: np.ndarray,
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create input/output pairs for autoregressive training.

        For next-token prediction:
        - Input: token_ids[:-1], positions[:-1]
        - Target: token_ids[1:]

        Args:
            token_ids: Shape (seq_len,).
            positions: Shape (seq_len, 4).

        Returns:
            Tuple of (input_ids, input_positions, target_ids, input_mask)
            where input_mask marks real tokens (not padding).
        """
        input_ids = token_ids[:-1]
        input_positions = positions[:-1]
        target_ids = token_ids[1:]
        mask = (input_ids != self.pad_id).astype(np.float32)

        return input_ids, input_positions, target_ids, mask


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_tokenizer():
    """Run comprehensive tests for the tokenizer."""
    print("Testing tokenizer...")

    tok = Tokenizer()

    # Test vocabulary
    assert tok.vocab_size > 0
    assert tok.pad_id == 0
    print(f"  Vocabulary size: {tok.vocab_size}")

    # Test encoding/decoding of special tokens
    for tag in ["[PAD]", "[INPUT]", "[SPLIT]", "[SUB_MUL_0]", "[SUB_MUL_1]",
                "[SUB_MUL_2]", "[ADD]", "[SUB]", "[COMBINE]", "[OUTPUT]",
                "[SEP]", "[BASE_MUL]"]:
        tid = tok.encode_token(tag)
        decoded = tok.decode_token(tid)
        assert decoded == tag, f"Round-trip failed: {tag} -> {tid} -> {decoded}"
    print("  Special token encode/decode: PASSED")

    # Test bit tokens
    assert tok.encode_token(0) == tok.encode_token("0")
    assert tok.encode_token(1) == tok.encode_token("1")
    assert tok.decode_token(tok.encode_token(0)) == "0"
    assert tok.decode_token(tok.encode_token(1)) == "1"
    print("  Bit token encode/decode: PASSED")

    # Test school partial tokens
    for i in range(8):
        tag = f"[PARTIAL_{i}]"
        tid = tok.encode_token(tag)
        assert tok.decode_token(tid) == tag
    print("  School partial tokens: PASSED")

    # Test sub-problem ID hashing
    assert tok._hash_sub_problem_id(()) == 0
    # Depth 1: (0,) (1,) (2,) should be 1, 2, 3
    assert tok._hash_sub_problem_id((0,)) == 1
    assert tok._hash_sub_problem_id((1,)) == 2
    assert tok._hash_sub_problem_id((2,)) == 3
    # Depth 2: (0,0) (0,1) (0,2) (1,0) ... should be 5, 6, 7, 8, ...
    id_00 = tok._hash_sub_problem_id((0, 0))
    id_01 = tok._hash_sub_problem_id((0, 1))
    id_02 = tok._hash_sub_problem_id((0, 2))
    id_10 = tok._hash_sub_problem_id((1, 0))
    assert id_00 != id_01 != id_02 != id_10, "Sub-problem IDs should be unique"
    assert id_00 > 3, "Depth-2 IDs should be > depth-1 IDs"
    print("  Sub-problem ID hashing: PASSED")

    # Test encoding a Karatsuba trace sequence
    from src.data.karatsuba_trace import KaratsubaTraceGenerator
    gen = KaratsubaTraceGenerator(base_case_bits=4)
    trace = gen.generate(7, 5, 4, ordering="depth_first")
    seq = gen.trace_to_token_sequence(trace)

    token_ids, positions = tok.encode_trace_sequence(seq)
    assert len(token_ids) == len(seq)
    assert positions.shape == (len(seq), 4)
    print(f"  Karatsuba 4-bit trace: {len(token_ids)} tokens encoded")

    # Test round-trip: decode and check
    decoded = tok.decode_token_ids(token_ids)
    for i, ((orig_tok, _), dec_tok) in enumerate(zip(seq, decoded)):
        if isinstance(orig_tok, int):
            assert dec_tok == str(orig_tok), (
                f"Mismatch at position {i}: {orig_tok} -> {dec_tok}"
            )
        else:
            assert dec_tok == orig_tok, (
                f"Mismatch at position {i}: {orig_tok} -> {dec_tok}"
            )
    print("  Token round-trip: PASSED")

    # Test position coupling
    coupling_ids = tok.compute_position_coupling_ids(positions)
    assert len(coupling_ids) == len(token_ids)

    # Verify: bit tokens with same significance and depth should have same coupling ID
    # (This is a structural test - find pairs of tokens that should be coupled)
    bit_0_positions = []
    for i in range(len(token_ids)):
        if positions[i, 0] == 0 and positions[i, 0] >= 0:  # bit significance 0
            bit_0_positions.append(i)
    if len(bit_0_positions) > 1:
        # All bit-0 tokens in the same sub-problem/depth should share coupling
        for j in range(1, len(bit_0_positions)):
            idx_a = bit_0_positions[0]
            idx_b = bit_0_positions[j]
            if (positions[idx_a, 1] == positions[idx_b, 1] and
                positions[idx_a, 2] == positions[idx_b, 2]):
                assert coupling_ids[idx_a] == coupling_ids[idx_b], (
                    f"Tokens at {idx_a} and {idx_b} should be coupled"
                )
    print("  Position coupling: PASSED")

    # Test padding
    padded_ids, padded_pos, mask = tok.pad_sequence(token_ids, positions, max_len=512)
    assert len(padded_ids) == 512
    assert padded_pos.shape == (512, 4)
    assert mask.sum() == len(token_ids)
    assert padded_ids[len(token_ids)] == PAD_ID
    print("  Padding: PASSED")

    # Test input/output pair creation
    inp_ids, inp_pos, tgt_ids, inp_mask = tok.create_input_output_pair(
        token_ids, positions
    )
    assert len(inp_ids) == len(token_ids) - 1
    assert len(tgt_ids) == len(token_ids) - 1
    # Target should be shifted by 1 from input
    for i in range(len(inp_ids)):
        assert tgt_ids[i] == token_ids[i + 1]
    print("  Input/output pairs: PASSED")

    # Test encoding a school trace
    from src.data.school_trace import SchoolTraceGenerator
    school_gen = SchoolTraceGenerator()
    school_trace = school_gen.generate(7, 5, 4)
    school_seq = school_gen.trace_to_token_sequence(school_trace)

    school_ids, school_pos = tok.encode_trace_sequence(school_seq)
    assert len(school_ids) == len(school_seq)
    print(f"  School 4-bit trace: {len(school_ids)} tokens encoded")

    # Test 8-bit Karatsuba trace
    gen8 = KaratsubaTraceGenerator(base_case_bits=4)
    trace8 = gen8.generate(173, 211, 8, ordering="depth_first")
    seq8 = gen8.trace_to_token_sequence(trace8)
    ids8, pos8 = tok.encode_trace_sequence(seq8)
    print(f"  Karatsuba 8-bit trace: {len(ids8)} tokens")

    print("\nAll tokenizer tests PASSED!")


if __name__ == "__main__":
    _test_tokenizer()
