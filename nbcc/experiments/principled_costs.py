"""
Principled cost models based on real CPU instruction characteristics.

This module provides cost models grounded in actual hardware behavior:
- Instruction latencies from CPU microarchitecture (Agner Fog's tables)
- Memory hierarchy effects (register, L1, L2, L3, RAM)
- Control flow costs (branch prediction, misprediction penalties)
- Call overhead (stack operations, register spills)

References:
- Agner Fog's Instruction Tables: https://agner.org/optimize/
- Intel 64 and IA-32 Optimization Reference Manual
- LLVM Cost Model: llvm/include/llvm/Analysis/TargetTransformInfo.h
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

from sealir.eqsat.rvsdg_extract import CostFunc, CostModel as _CostModel


class CostCategory(IntEnum):
    """Cost categories for operation classification."""
    ZERO = 0          # Eliminated by compiler (constants, type casts)
    TRIVIAL = 1       # Single cycle (register moves, simple ALU)
    SIMPLE_ALU = 2    # 1-3 cycles (add, sub, compare, bitwise)
    COMPLEX_ALU = 3   # 3-10 cycles (mul, shift)
    DIVISION = 4      # 10-40 cycles (div, mod)
    MEMORY_L1 = 5     # ~4 cycles (L1 cache hit)
    MEMORY_L2 = 6     # ~12 cycles (L2 cache hit)
    MEMORY_L3 = 7     # ~40 cycles (L3 cache hit)
    MEMORY_RAM = 8    # ~200 cycles (RAM access)
    BRANCH = 9        # 1-15 cycles (depends on prediction)
    CALL = 10         # 2-5 cycles (direct call)
    INDIRECT_CALL = 11  # 5-20 cycles (virtual/indirect call)
    SYSCALL = 12      # 100+ cycles (OS interaction)


@dataclass(frozen=True)
class InstructionCost:
    """
    CPU instruction cost model based on microarchitecture.

    Values are approximate cycles for modern x86-64 (Skylake-era).
    These represent throughput-limited costs (not latency).
    """
    # Zero cost - eliminated at compile time
    CONSTANT: float = 0.0
    TYPE_CAST: float = 0.0  # Same-size casts are free

    # Trivial - 1 cycle
    MOVE: float = 0.25      # Register move (often eliminated)
    NOP: float = 0.0

    # Simple ALU - 1 cycle latency, 0.25-0.5 throughput
    ADD_INT: float = 0.5
    SUB_INT: float = 0.5
    AND_INT: float = 0.5
    OR_INT: float = 0.5
    XOR_INT: float = 0.5
    NOT_INT: float = 0.5
    CMP_INT: float = 0.5

    # Shifts - 1 cycle
    SHIFT: float = 0.5

    # Integer multiply - 3-4 cycles latency, 1 throughput
    MUL_INT: float = 1.0

    # Integer division - 20-100 cycles (data dependent)
    DIV_INT: float = 25.0
    MOD_INT: float = 25.0

    # Floating point - varies by operation
    ADD_FP: float = 0.5     # 4 cycle latency, 0.5 throughput
    SUB_FP: float = 0.5
    MUL_FP: float = 0.5     # 4 cycle latency, 0.5 throughput
    DIV_FP: float = 5.0     # 11-14 cycles
    SQRT_FP: float = 6.0    # 12-18 cycles

    # Comparisons
    CMP_FP: float = 0.5

    # Memory access (assuming good cache behavior)
    LOAD_L1: float = 4.0
    STORE_L1: float = 1.0

    # Control flow
    BRANCH_PREDICTED: float = 0.5    # Well-predicted branch
    BRANCH_MISPREDICTED: float = 15.0  # Misprediction penalty
    BRANCH_AVG: float = 2.0          # Average (90% prediction rate)

    # Function calls
    CALL_DIRECT: float = 3.0
    CALL_INDIRECT: float = 10.0
    RETURN: float = 1.0

    # Python/dynamic dispatch (interpreted, not JIT)
    PYTHON_DISPATCH: float = 100.0   # Dynamic lookup + call
    PYTHON_GLOBAL: float = 50.0      # Global variable lookup


# Operation to instruction cost mapping
OPERATION_COSTS: dict[str, float] = {
    # Integer arithmetic
    "Op_i32_add": InstructionCost.ADD_INT,
    "Op_i32_sub": InstructionCost.SUB_INT,
    "Op_i32_mul": InstructionCost.MUL_INT,
    "Op_i32_div": InstructionCost.DIV_INT,
    "Op_i32_mod": InstructionCost.MOD_INT,
    "Op_i32_neg": InstructionCost.SUB_INT,  # neg = 0 - x

    # Integer comparisons
    "Op_i32_lt": InstructionCost.CMP_INT,
    "Op_i32_le": InstructionCost.CMP_INT,
    "Op_i32_gt": InstructionCost.CMP_INT,
    "Op_i32_ge": InstructionCost.CMP_INT,
    "Op_i32_eq": InstructionCost.CMP_INT,
    "Op_i32_ne": InstructionCost.CMP_INT,

    # Integer bitwise
    "Op_i32_and": InstructionCost.AND_INT,
    "Op_i32_or": InstructionCost.OR_INT,
    "Op_i32_xor": InstructionCost.XOR_INT,
    "Op_i32_lshift": InstructionCost.SHIFT,
    "Op_i32_rshift": InstructionCost.SHIFT,

    # Float arithmetic
    "Op_f64_add": InstructionCost.ADD_FP,
    "Op_f64_sub": InstructionCost.SUB_FP,
    "Op_f64_mul": InstructionCost.MUL_FP,
    "Op_f64_div": InstructionCost.DIV_FP,
    "Op_f64_neg": InstructionCost.SUB_FP,

    # Float comparisons
    "Op_f64_lt": InstructionCost.CMP_FP,
    "Op_f64_le": InstructionCost.CMP_FP,
    "Op_f64_gt": InstructionCost.CMP_FP,
    "Op_f64_ge": InstructionCost.CMP_FP,
    "Op_f64_eq": InstructionCost.CMP_FP,
    "Op_f64_ne": InstructionCost.CMP_FP,

    # Boolean operations
    "Op_bool_and": InstructionCost.AND_INT,
    "Op_bool_or": InstructionCost.OR_INT,
    "Op_bool_not": InstructionCost.NOT_INT,

    # Specialized calls (inlined or direct)
    "CallFQN": InstructionCost.CALL_DIRECT,

    # Python operations (expensive - dynamic dispatch)
    "Py_Call": InstructionCost.PYTHON_DISPATCH,
    "Py_LoadGlobal": InstructionCost.PYTHON_GLOBAL,

    # IO operations (syscall-level)
    "Builtin_print_i32": 500.0,  # Syscall + formatting
    "Builtin_print_f64": 500.0,
    "Builtin_print_str": 500.0,

    # Constants (free)
    "Literal": InstructionCost.CONSTANT,
    "Const": InstructionCost.CONSTANT,
}

# Control flow costs (base + per-iteration for loops)
CONTROL_FLOW_COSTS: dict[str, tuple[float, float]] = {
    # (base_cost, per_child_multiplier)
    "Gamma": (InstructionCost.BRANCH_AVG, 0.5),  # Conditional: one branch taken
    "Theta": (InstructionCost.BRANCH_AVG, 1.0),  # Loop: branch per iteration
}


class InstructionCostModel(_CostModel):
    """
    Cost model based on real CPU instruction latencies.

    This model uses measured/documented instruction costs from CPU vendors
    and optimization guides. It accounts for:

    - Instruction throughput (not just latency)
    - Simple vs complex ALU operations
    - Division as expensive operation
    - Branch prediction effects
    - Dynamic dispatch overhead

    The key insight is that the extraction should prefer:
    1. Lowered operations (Op_*) over Python dispatch (Py_*)
    2. Additions/comparisons over divisions
    3. Direct calls over indirect calls
    """

    def __init__(self, scale_factor: float = 1.0):
        """
        Args:
            scale_factor: Multiplier for all costs (for calibration)
        """
        self.scale_factor = scale_factor
        self._call_count = 0
        self._op_histogram: dict[str, int] = {}

    def get_cost_function(
        self,
        nodename: str,
        op: str,
        ty: str,
        cost: float,
        children: Sequence[str],
    ) -> CostFunc:
        self._call_count += 1
        self._op_histogram[op] = self._op_histogram.get(op, 0) + 1

        # Look up base cost
        if op in OPERATION_COSTS:
            base_cost = OPERATION_COSTS[op] * self.scale_factor
            return self.get_simple(base_cost)

        # Control flow with child cost propagation
        if op in CONTROL_FLOW_COSTS:
            base, multiplier = CONTROL_FLOW_COSTS[op]
            scaled_base = base * self.scale_factor
            # Children costs are multiplied by the given factor
            multipliers = tuple([multiplier] * len(children))
            return self.get_scaled(scaled_base, multipliers)

        # Default: use parent implementation with small base cost
        return super().get_cost_function(nodename, op, ty, cost, children)

    def get_statistics(self) -> dict:
        """Return collected statistics."""
        return {
            "call_count": self._call_count,
            "op_histogram": dict(self._op_histogram),
        }

    def reset_statistics(self) -> None:
        """Reset collected statistics."""
        self._call_count = 0
        self._op_histogram = {}


@dataclass
class SubtreeCostAnalysis:
    """
    Analyzes subtree properties for more accurate cost estimation.

    Beyond local instruction costs, this considers:
    - Critical path length (longest dependency chain)
    - Parallelism opportunities (independent operations)
    - Memory access patterns
    - Loop nesting depth
    """
    total_nodes: int = 0
    arithmetic_ops: int = 0
    division_ops: int = 0
    comparison_ops: int = 0
    branch_ops: int = 0
    loop_ops: int = 0
    call_ops: int = 0
    python_ops: int = 0
    io_ops: int = 0
    max_depth: int = 0

    def estimated_cycles(self) -> float:
        """Estimate total cycles based on operation mix."""
        cycles = 0.0

        # Fast operations (can be parallelized)
        cycles += self.arithmetic_ops * 0.5
        cycles += self.comparison_ops * 0.5

        # Slow operations (serializing)
        cycles += self.division_ops * 25.0

        # Control flow
        cycles += self.branch_ops * 2.0
        cycles += self.loop_ops * 5.0  # Loop overhead

        # Calls
        cycles += self.call_ops * 3.0
        cycles += self.python_ops * 100.0

        # IO (dominates)
        cycles += self.io_ops * 500.0

        return cycles


class AnalyticalCostModel(_CostModel):
    """
    Cost model using analytical formulas for subtree costs.

    This model computes costs based on:
    1. Local instruction cost
    2. Data dependency chains (critical path)
    3. Execution model (in-order vs out-of-order)

    For extraction, the key property is that costs should be
    monotonically increasing with subtree complexity.
    """

    # Instruction Level Parallelism factor (how many ops can execute per cycle)
    ILP_FACTOR: float = 4.0

    # Categories and their base costs
    CATEGORY_COSTS: dict[str, float] = {
        "arithmetic": 0.5 / 4.0,  # Parallelizable
        "division": 25.0,         # Serializing
        "comparison": 0.5 / 4.0,  # Parallelizable
        "branch": 2.0,            # Some misprediction
        "loop": 5.0,              # Overhead per iteration
        "call": 3.0,              # Direct call
        "python": 100.0,          # Dynamic dispatch
        "io": 500.0,              # Syscall
        "constant": 0.0,          # Free
    }

    def _categorize_op(self, op: str) -> str:
        """Categorize operation for cost computation."""
        if op.startswith("Op_") and ("div" in op or "mod" in op):
            return "division"
        elif op.startswith("Op_") and any(c in op for c in ["lt", "le", "gt", "ge", "eq", "ne"]):
            return "comparison"
        elif op.startswith("Op_"):
            return "arithmetic"
        elif op == "Gamma":
            return "branch"
        elif op == "Theta":
            return "loop"
        elif op == "CallFQN":
            return "call"
        elif op.startswith("Py_"):
            return "python"
        elif op.startswith("Builtin_print"):
            return "io"
        elif op in ["Literal", "Const"]:
            return "constant"
        else:
            return "arithmetic"  # Default

    def get_cost_function(
        self,
        nodename: str,
        op: str,
        ty: str,
        cost: float,
        children: Sequence[str],
    ) -> CostFunc:
        category = self._categorize_op(op)
        base_cost = self.CATEGORY_COSTS.get(category, 1.0)

        # For control flow, propagate child costs
        if category == "branch":
            # Conditional: expect to take one branch
            # Cost = branch + max(then_cost, else_cost) approximately
            # Use average of children costs
            return self.get_scaled(base_cost, tuple([0.5] * len(children)))
        elif category == "loop":
            # Loop: cost = overhead + iterations * body_cost
            # Without knowing iterations, use multiplier
            return self.get_scaled(base_cost, tuple([10.0] * len(children)))
        else:
            # Regular ops: just base cost
            return self.get_simple(base_cost)


class CalibratedCostModel(_CostModel):
    """
    Cost model that can be calibrated from measured data.

    Starts with analytical costs but can be updated with
    actual measurements to improve accuracy.

    Calibration process:
    1. Run programs with analytical costs
    2. Measure actual runtime
    3. Compute scaling factors per operation category
    4. Update costs to better match reality
    """

    def __init__(self):
        # Start with analytical costs
        self._base_costs = dict(OPERATION_COSTS)
        self._calibration_factors: dict[str, float] = {}
        self._measurements: list[tuple[str, float, float]] = []  # (op, predicted, actual)

    def get_cost_function(
        self,
        nodename: str,
        op: str,
        ty: str,
        cost: float,
        children: Sequence[str],
    ) -> CostFunc:
        base = self._base_costs.get(op, 1.0)
        factor = self._calibration_factors.get(op, 1.0)
        calibrated = base * factor

        if op in CONTROL_FLOW_COSTS:
            _, multiplier = CONTROL_FLOW_COSTS[op]
            return self.get_scaled(calibrated, tuple([multiplier] * len(children)))

        return self.get_simple(calibrated)

    def add_measurement(self, op: str, predicted: float, actual: float) -> None:
        """Add a measurement for calibration."""
        self._measurements.append((op, predicted, actual))

    def calibrate(self) -> dict[str, float]:
        """
        Compute calibration factors from measurements.

        Uses least squares to find scaling factors that minimize
        the difference between predicted and actual costs.

        Returns:
            Dict mapping operation types to their calibration factors
        """
        from collections import defaultdict

        # Group measurements by operation
        op_data: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for op, pred, actual in self._measurements:
            op_data[op].append((pred, actual))

        # Compute factor for each operation
        factors = {}
        for op, data in op_data.items():
            if not data:
                continue

            # Simple: factor = mean(actual) / mean(predicted)
            sum_pred = sum(p for p, _ in data)
            sum_actual = sum(a for _, a in data)

            if sum_pred > 0:
                factors[op] = sum_actual / sum_pred
            else:
                factors[op] = 1.0

        self._calibration_factors = factors
        return factors

    def save_calibration(self, path: str) -> None:
        """Save calibration data to file."""
        import json
        data = {
            "calibration_factors": self._calibration_factors,
            "measurements": self._measurements,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_calibration(self, path: str) -> None:
        """Load calibration data from file."""
        import json
        with open(path) as f:
            data = json.load(f)
        self._calibration_factors = data.get("calibration_factors", {})
        self._measurements = [tuple(m) for m in data.get("measurements", [])]


def create_cost_comparison_table() -> str:
    """Generate a table comparing operation costs across models."""
    ops = sorted(OPERATION_COSTS.keys())

    lines = [
        "Operation Cost Comparison",
        "=" * 60,
        f"{'Operation':<25} {'Cycles':>10} {'Category':>15}",
        "-" * 60,
    ]

    # Categorize by cost
    trivial = []
    simple = []
    medium = []
    expensive = []
    very_expensive = []

    for op in ops:
        cost = OPERATION_COSTS[op]
        if cost == 0:
            trivial.append((op, cost))
        elif cost <= 1:
            simple.append((op, cost))
        elif cost <= 5:
            medium.append((op, cost))
        elif cost <= 50:
            expensive.append((op, cost))
        else:
            very_expensive.append((op, cost))

    for category, items in [
        ("Free (0 cycles)", trivial),
        ("Trivial (<1 cycle)", simple),
        ("Simple (1-5 cycles)", medium),
        ("Expensive (5-50 cycles)", expensive),
        ("Very Expensive (>50 cycles)", very_expensive),
    ]:
        if items:
            lines.append(f"\n{category}:")
            for op, cost in sorted(items, key=lambda x: x[1]):
                lines.append(f"  {op:<23} {cost:>10.1f}")

    return "\n".join(lines)
