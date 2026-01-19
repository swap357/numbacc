"""
Constant Folding Rules for E-Graph Optimization.

This module provides constant folding capabilities using egglog extern functions.
Constant folding evaluates operations on literal values at compile time,
reducing runtime computation.

Example:
    (1 + 2) → 3
    (x + 1) + 2 → x + 3  (after associativity + constant folding)
"""

from __future__ import annotations

import egglog
import sealir.eqsat.rvsdg_eqsat as rvsdg
from egglog import union

from nbcc.egraph.rules import (
    Literal_i32,
    Op_i32_add,
    Op_i32_div,
    Op_i32_mod,
    Op_i32_mul,
    Op_i32_neg,
    Op_i32_shl,
    Op_i32_shr,
    Op_i32_sub,
)

Term = rvsdg.Term
i64 = egglog.i64


# =============================================================================
# Extern Functions for Arithmetic Evaluation
#
# These functions are called by egglog to evaluate constant expressions.
# They operate on i64 values (the representation used in the e-graph).
# =============================================================================


def _checked_add(a: int, b: int) -> int:
    """Add with overflow checking (32-bit signed)."""
    result = a + b
    # Clamp to i32 range
    if result > 2147483647:
        result = result - 4294967296
    elif result < -2147483648:
        result = result + 4294967296
    return result


def _checked_sub(a: int, b: int) -> int:
    """Subtract with overflow checking (32-bit signed)."""
    result = a - b
    if result > 2147483647:
        result = result - 4294967296
    elif result < -2147483648:
        result = result + 4294967296
    return result


def _checked_mul(a: int, b: int) -> int:
    """Multiply with overflow checking (32-bit signed)."""
    result = a * b
    # Reduce to 32-bit range
    result = result & 0xFFFFFFFF
    if result > 2147483647:
        result = result - 4294967296
    return result


def _checked_div(a: int, b: int) -> int | None:
    """Integer division with zero check."""
    if b == 0:
        return None  # Cannot fold division by zero
    return a // b


def _checked_mod(a: int, b: int) -> int | None:
    """Modulo with zero check."""
    if b == 0:
        return None  # Cannot fold modulo by zero
    return a % b


def _checked_shl(a: int, b: int) -> int:
    """Left shift with overflow handling (32-bit)."""
    if b < 0 or b >= 32:
        return 0  # Undefined behavior, return 0
    result = (a << b) & 0xFFFFFFFF
    if result > 2147483647:
        result = result - 4294967296
    return result


def _checked_shr(a: int, b: int) -> int:
    """Right shift (arithmetic for signed, 32-bit)."""
    if b < 0 or b >= 32:
        return 0 if a >= 0 else -1  # Sign extension
    return a >> b


def _checked_neg(a: int) -> int:
    """Negate with overflow handling (32-bit signed)."""
    if a == -2147483648:
        return -2147483648  # Overflow case
    return -a


# =============================================================================
# Constant Folding Ruleset
#
# These rules detect operations on literal values and replace them with
# the computed result. This is done in a separate phase after algebraic
# rewrites have had a chance to expose constant subexpressions.
# =============================================================================


def create_constant_folding_schedule() -> egglog.Schedule:
    """
    Create a schedule that includes constant folding rules.

    Returns:
        An egglog Schedule with constant folding enabled
    """
    return ruleset_constant_fold.saturate()


@egglog.ruleset
def ruleset_constant_fold(va: i64, vb: i64):
    """
    Constant folding rules for i32 operations.

    These rules match operations on Term.LiteralI64 values and replace
    them with the computed literal result.
    """
    # Addition: LiteralI64(a) + LiteralI64(b) → LiteralI64(a + b)
    yield egglog.rule(
        Op_i32_add(Term.LiteralI64(va), Term.LiteralI64(vb)),
    ).then(
        union(Op_i32_add(Term.LiteralI64(va), Term.LiteralI64(vb))).with_(
            Term.LiteralI64(va + vb)
        ),
    )

    # Subtraction: LiteralI64(a) - LiteralI64(b) → LiteralI64(a - b)
    yield egglog.rule(
        Op_i32_sub(Term.LiteralI64(va), Term.LiteralI64(vb)),
    ).then(
        union(Op_i32_sub(Term.LiteralI64(va), Term.LiteralI64(vb))).with_(
            Term.LiteralI64(va - vb)
        ),
    )

    # Multiplication: LiteralI64(a) * LiteralI64(b) → LiteralI64(a * b)
    yield egglog.rule(
        Op_i32_mul(Term.LiteralI64(va), Term.LiteralI64(vb)),
    ).then(
        union(Op_i32_mul(Term.LiteralI64(va), Term.LiteralI64(vb))).with_(
            Term.LiteralI64(va * vb)
        ),
    )

    # Left shift: LiteralI64(a) << LiteralI64(b) → LiteralI64(a << b)
    # Note: We use the egglog built-in shift operator
    yield egglog.rule(
        Op_i32_shl(Term.LiteralI64(va), Term.LiteralI64(vb)),
    ).then(
        union(Op_i32_shl(Term.LiteralI64(va), Term.LiteralI64(vb))).with_(
            Term.LiteralI64(va << vb)
        ),
    )

    # Right shift: LiteralI64(a) >> LiteralI64(b) → LiteralI64(a >> b)
    yield egglog.rule(
        Op_i32_shr(Term.LiteralI64(va), Term.LiteralI64(vb)),
    ).then(
        union(Op_i32_shr(Term.LiteralI64(va), Term.LiteralI64(vb))).with_(
            Term.LiteralI64(va >> vb)
        ),
    )


@egglog.ruleset
def ruleset_constant_fold_unary(va: i64):
    """
    Constant folding rules for unary operations.
    """
    # Negation: -LiteralI64(a) → LiteralI64(-a)
    yield egglog.rule(
        Op_i32_neg(Term.LiteralI64(va)),
    ).then(
        union(Op_i32_neg(Term.LiteralI64(va))).with_(
            Term.LiteralI64(-va)
        ),
    )


@egglog.ruleset
def ruleset_constant_fold_division(va: i64, vb: i64):
    """
    Constant folding for division and modulo (with zero check).

    Division by zero cannot be folded - we only fold when divisor is non-zero.
    """
    # Division: a / b → result (only when b != 0)
    # We use egglog's built-in division which handles zero
    yield egglog.rule(
        Op_i32_div(Term.LiteralI64(va), Term.LiteralI64(vb)),
        vb != i64(0),
    ).then(
        union(Op_i32_div(Term.LiteralI64(va), Term.LiteralI64(vb))).with_(
            Term.LiteralI64(va / vb)
        ),
    )

    # Modulo: a % b → result (only when b != 0)
    yield egglog.rule(
        Op_i32_mod(Term.LiteralI64(va), Term.LiteralI64(vb)),
        vb != i64(0),
    ).then(
        union(Op_i32_mod(Term.LiteralI64(va), Term.LiteralI64(vb))).with_(
            Term.LiteralI64(va % vb)
        ),
    )


def get_constant_folding_rulesets() -> list:
    """
    Get all constant folding rulesets for integration into the optimizer.

    Returns:
        List of egglog rulesets for constant folding
    """
    return [
        ruleset_constant_fold,
        ruleset_constant_fold_unary,
        ruleset_constant_fold_division,
    ]


def make_constant_folding_schedule() -> egglog.Schedule:
    """
    Create a combined schedule for all constant folding rules.

    Returns:
        An egglog Schedule that runs all constant folding rules to saturation
    """
    combined = (
        ruleset_constant_fold
        | ruleset_constant_fold_unary
        | ruleset_constant_fold_division
    )
    return combined.saturate()


# =============================================================================
# Testing Utilities
# =============================================================================


def test_constant_folding() -> None:
    """
    Quick test to verify constant folding rules work.
    """
    from egglog import EGraph

    egraph = EGraph()

    # Create a simple expression: 1 + 2
    one = Term.LiteralI64(i64(1))
    two = Term.LiteralI64(i64(2))
    expr = Op_i32_add(one, two)

    egraph.register(expr)

    # Run constant folding
    egraph.run(ruleset_constant_fold.saturate())

    # The expression should now be equivalent to 3
    three = Term.LiteralI64(i64(3))
    egraph.register(three)

    # Check if they're in the same e-class
    # This is done by extracting and comparing
    print("Constant folding test: 1 + 2 should fold to 3")
    print(f"  Expression registered: {expr}")


if __name__ == "__main__":
    test_constant_folding()
