"""
Synthetic data generation for cost model training.

Since we can't always measure actual runtime, this module generates
synthetic ground truth based on analytical cost models. The idea is:

1. Generate programs with known operation mixes
2. Compute "expected" cost using analytical model
3. Use this as training signal for ML model

This allows the ML model to learn the analytical cost structure,
which can then be refined with real measurements when available.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .principled_costs import OPERATION_COSTS, InstructionCost


@dataclass
class SyntheticProgram:
    """A synthetically generated program with known cost characteristics."""
    name: str
    source: str
    operation_counts: dict[str, int]
    expected_cost: float
    depth: int
    has_loops: bool
    has_branches: bool


@dataclass
class OperationMix:
    """Defines the mix of operations in a synthetic program."""
    # Arithmetic operations
    additions: int = 0
    subtractions: int = 0
    multiplications: int = 0
    divisions: int = 0

    # Comparisons
    comparisons: int = 0

    # Control flow
    branches: int = 0
    loops: int = 0
    loop_iterations: int = 10  # Default iteration count

    # Calls
    function_calls: int = 0

    # IO
    prints: int = 0

    def expected_cycles(self) -> float:
        """Compute expected cycles based on instruction costs."""
        cycles = 0.0

        # Arithmetic
        cycles += self.additions * InstructionCost.ADD_INT
        cycles += self.subtractions * InstructionCost.SUB_INT
        cycles += self.multiplications * InstructionCost.MUL_INT
        cycles += self.divisions * InstructionCost.DIV_INT

        # Comparisons
        cycles += self.comparisons * InstructionCost.CMP_INT

        # Control flow
        cycles += self.branches * InstructionCost.BRANCH_AVG
        cycles += self.loops * InstructionCost.BRANCH_AVG * self.loop_iterations

        # Calls
        cycles += self.function_calls * InstructionCost.CALL_DIRECT

        # IO (very expensive)
        cycles += self.prints * 500.0

        return cycles

    def to_dict(self) -> dict[str, int]:
        """Convert to operation count dictionary."""
        return {
            "Op_i32_add": self.additions,
            "Op_i32_sub": self.subtractions,
            "Op_i32_mul": self.multiplications,
            "Op_i32_div": self.divisions,
            "Op_i32_lt": self.comparisons,  # Use lt as representative
            "Gamma": self.branches,
            "Theta": self.loops,
            "CallFQN": self.function_calls,
            "Builtin_print_i32": self.prints,
        }


def generate_arithmetic_chain(length: int, ops: list[str] | None = None) -> tuple[str, OperationMix]:
    """
    Generate a chain of arithmetic operations.

    Example output for length=3, ops=['+', '*']:
        y = x + 1
        y = y * 2
        y = y + 3
    """
    if ops is None:
        ops = ["+", "-", "*"]

    lines = []
    mix = OperationMix()

    lines.append(f"def chain_{length}(x: i32) -> i32:")
    lines.append("    y = x")

    for i in range(length):
        op = random.choice(ops)
        val = random.randint(1, 10)

        if op == "+":
            lines.append(f"    y = y + {val}")
            mix.additions += 1
        elif op == "-":
            lines.append(f"    y = y - {val}")
            mix.subtractions += 1
        elif op == "*":
            lines.append(f"    y = y * {val}")
            mix.multiplications += 1
        elif op == "//":
            lines.append(f"    y = y // {val}")
            mix.divisions += 1

    lines.append("    return y")
    lines.append("")

    return "\n".join(lines), mix


def generate_branch_tree(depth: int, balanced: bool = True) -> tuple[str, OperationMix]:
    """
    Generate nested conditionals.

    Example output for depth=2:
        if x > 0:
            if x > 10:
                result = x * 2
            else:
                result = x + 1
        else:
            result = 0 - x
    """
    mix = OperationMix()

    def indent(level: int) -> str:
        return "    " * (level + 1)

    def generate_level(level: int, var: str) -> list[str]:
        if level >= depth:
            # Leaf: simple operation
            mix.additions += 1
            return [f"{indent(level)}result = {var} + 1"]

        mix.comparisons += 1
        mix.branches += 1

        lines = []
        threshold = random.randint(1, 10) * (2 ** level)

        lines.append(f"{indent(level)}if {var} > {threshold}:")
        lines.extend(generate_level(level + 1, var))

        if balanced:
            lines.append(f"{indent(level)}else:")
            lines.extend(generate_level(level + 1, var))

        return lines

    func_lines = [f"def branch_tree_{depth}(x: i32) -> i32:"]
    func_lines.append("    result = 0")
    func_lines.extend(generate_level(0, "x"))
    func_lines.append("    return result")
    func_lines.append("")

    return "\n".join(func_lines), mix


def generate_loop(iterations_expr: str, body_ops: int) -> tuple[str, OperationMix]:
    """
    Generate a loop with specified body complexity.

    Example output:
        total = 0
        i = 0
        while i < n:
            total = total + i * 2
            i = i + 1
        return total
    """
    mix = OperationMix()
    mix.loops = 1
    mix.comparisons = 1  # Loop condition

    lines = [f"def loop_{body_ops}(n: i32) -> i32:"]
    lines.append("    total = 0")
    lines.append("    i = 0")
    lines.append(f"    while i < {iterations_expr}:")

    # Generate body operations
    for j in range(body_ops):
        op = random.choice(["+", "*"])
        if op == "+":
            lines.append(f"        total = total + i")
            mix.additions += 1
        else:
            lines.append(f"        total = total * 2")
            mix.multiplications += 1

    # Loop increment
    lines.append("        i = i + 1")
    mix.additions += 1

    lines.append("    return total")
    lines.append("")

    return "\n".join(lines), mix


def generate_division_heavy(count: int) -> tuple[str, OperationMix]:
    """
    Generate program heavy in divisions (expensive operations).

    This tests whether the cost model properly penalizes divisions.
    """
    mix = OperationMix()
    mix.divisions = count

    lines = [f"def div_heavy_{count}(x: i32) -> i32:"]
    lines.append("    y = x")

    for i in range(count):
        divisor = random.randint(2, 10)
        lines.append(f"    y = y // {divisor}")

    lines.append("    return y")
    lines.append("")

    return "\n".join(lines), mix


def generate_mixed_program(
    arithmetic: int = 5,
    branches: int = 2,
    loops: int = 1,
    divisions: int = 0,
) -> tuple[str, OperationMix]:
    """Generate a program with a specific mix of operations."""
    mix = OperationMix()
    lines = ["def mixed(x: i32) -> i32:"]
    lines.append("    result = x")

    # Arithmetic chain
    for i in range(arithmetic):
        op = random.choice(["+", "-", "*"])
        val = random.randint(1, 5)
        if op == "+":
            lines.append(f"    result = result + {val}")
            mix.additions += 1
        elif op == "-":
            lines.append(f"    result = result - {val}")
            mix.subtractions += 1
        else:
            lines.append(f"    result = result * {val}")
            mix.multiplications += 1

    # Divisions
    for i in range(divisions):
        lines.append(f"    result = result // {random.randint(2, 5)}")
        mix.divisions += 1

    # Branches
    indent = "    "
    for i in range(branches):
        thresh = random.randint(1, 100)
        lines.append(f"{indent}if result > {thresh}:")
        lines.append(f"{indent}    result = result + 1")
        mix.comparisons += 1
        mix.branches += 1
        mix.additions += 1

    # Loops
    for i in range(loops):
        lines.append(f"    i = 0")
        lines.append(f"    while i < 10:")
        lines.append(f"        result = result + i")
        lines.append(f"        i = i + 1")
        mix.loops += 1
        mix.comparisons += 1
        mix.additions += 2

    lines.append("    return result")
    lines.append("")

    return "\n".join(lines), mix


class SyntheticDataGenerator:
    """
    Generates a suite of synthetic programs for cost model training.

    The generator creates programs with:
    - Varying operation mixes
    - Known expected costs
    - Different complexity levels

    This provides training data for ML cost models.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def generate_training_suite(
        self,
        num_samples: int = 100,
    ) -> list[SyntheticProgram]:
        """Generate a diverse suite of training programs."""
        programs = []

        # Arithmetic chains of varying length
        for length in [5, 10, 20, 50]:
            source, mix = generate_arithmetic_chain(length)
            programs.append(SyntheticProgram(
                name=f"arith_chain_{length}",
                source=source,
                operation_counts=mix.to_dict(),
                expected_cost=mix.expected_cycles(),
                depth=1,
                has_loops=False,
                has_branches=False,
            ))

        # Branch trees of varying depth
        for depth in [1, 2, 3, 4]:
            source, mix = generate_branch_tree(depth)
            programs.append(SyntheticProgram(
                name=f"branch_tree_{depth}",
                source=source,
                operation_counts=mix.to_dict(),
                expected_cost=mix.expected_cycles(),
                depth=depth,
                has_loops=False,
                has_branches=True,
            ))

        # Loops with varying body complexity
        for body_ops in [1, 3, 5, 10]:
            source, mix = generate_loop("n", body_ops)
            programs.append(SyntheticProgram(
                name=f"loop_body_{body_ops}",
                source=source,
                operation_counts=mix.to_dict(),
                expected_cost=mix.expected_cycles(),
                depth=2,
                has_loops=True,
                has_branches=False,
            ))

        # Division-heavy programs
        for count in [1, 5, 10]:
            source, mix = generate_division_heavy(count)
            programs.append(SyntheticProgram(
                name=f"div_heavy_{count}",
                source=source,
                operation_counts=mix.to_dict(),
                expected_cost=mix.expected_cycles(),
                depth=1,
                has_loops=False,
                has_branches=False,
            ))

        # Random mixed programs
        random.seed(self.seed)
        for i in range(num_samples - len(programs)):
            arith = random.randint(1, 20)
            branches = random.randint(0, 3)
            loops = random.randint(0, 2)
            divs = random.randint(0, 5)

            source, mix = generate_mixed_program(arith, branches, loops, divs)
            programs.append(SyntheticProgram(
                name=f"mixed_{i}",
                source=source,
                operation_counts=mix.to_dict(),
                expected_cost=mix.expected_cycles(),
                depth=1 + branches + loops,
                has_loops=loops > 0,
                has_branches=branches > 0,
            ))

        return programs

    def save_suite(self, programs: list[SyntheticProgram], output_dir: Path) -> None:
        """Save generated programs to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each program
        for prog in programs:
            prog_file = output_dir / f"{prog.name}.spy"
            with open(prog_file, "w") as f:
                f.write(f'"""\nSynthetic program: {prog.name}\n')
                f.write(f"Expected cost: {prog.expected_cost:.2f} cycles\n")
                f.write(f"Operations: {prog.operation_counts}\n")
                f.write('"""\n\n')
                f.write(prog.source)
                f.write("\ndef main() -> i32:\n")
                # Find the function name
                func_name = prog.source.split("(")[0].replace("def ", "")
                f.write(f"    result = {func_name}(42)\n")
                f.write("    print(result)\n")
                f.write("    return 0\n")

        # Save manifest
        manifest = {
            "programs": [
                {
                    "name": p.name,
                    "expected_cost": p.expected_cost,
                    "operation_counts": p.operation_counts,
                    "has_loops": p.has_loops,
                    "has_branches": p.has_branches,
                }
                for p in programs
            ]
        }

        import json
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)


def generate_training_data(
    num_samples: int = 100,
    output_dir: str | Path | None = None,
) -> list[SyntheticProgram]:
    """
    Generate training data for ML cost model.

    Args:
        num_samples: Number of synthetic programs to generate
        output_dir: Optional directory to save programs

    Returns:
        List of SyntheticProgram with source code and expected costs
    """
    gen = SyntheticDataGenerator()
    programs = gen.generate_training_suite(num_samples)

    if output_dir:
        gen.save_suite(programs, Path(output_dir))

    return programs


def programs_to_features_and_costs(
    programs: list[SyntheticProgram],
) -> tuple[list[list[float]], list[float]]:
    """
    Convert synthetic programs to ML training format.

    Returns:
        Tuple of (features, costs) where:
        - features: List of feature vectors
        - costs: List of expected costs
    """
    features = []
    costs = []

    for prog in programs:
        # Feature vector: operation counts + metadata
        feature = [
            prog.operation_counts.get("Op_i32_add", 0),
            prog.operation_counts.get("Op_i32_sub", 0),
            prog.operation_counts.get("Op_i32_mul", 0),
            prog.operation_counts.get("Op_i32_div", 0),
            prog.operation_counts.get("Op_i32_lt", 0),
            prog.operation_counts.get("Gamma", 0),
            prog.operation_counts.get("Theta", 0),
            prog.operation_counts.get("CallFQN", 0),
            prog.operation_counts.get("Builtin_print_i32", 0),
            float(prog.depth),
            float(prog.has_loops),
            float(prog.has_branches),
        ]
        features.append(feature)
        costs.append(prog.expected_cost)

    return features, costs
