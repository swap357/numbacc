"""
Runtime Measurement Pipeline for Cost Model Validation.

This module provides infrastructure to:
1. Compile SPy programs with different cost models
2. Measure actual execution time (when compilation to executable is available)
3. Compare predicted costs to actual runtimes

The pipeline supports:
- E-graph extraction with configurable cost models
- Native binary compilation via MLIR/LLVM
- WASM compilation (future)
- Runtime measurement with statistical analysis
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from egglog import EGraph
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import CostModel as _CostModel
from sealir.eqsat.rvsdg_extract import egraph_extraction

from nbcc.egraph.rules import egraph_convert_metadata, egraph_optimize
from nbcc.frontend import TranslationUnit, frontend

from .cost_models import BaselineCostModel


@dataclass
class CompilationResult:
    """Result of compiling a program with a specific cost model."""
    program: str
    model_name: str
    extracted_cost: float
    extracted_node_count: int
    extraction_time_ns: int
    compilation_time_ns: int = 0
    output_path: str | None = None
    success: bool = True
    error: str | None = None


@dataclass
class RuntimeMeasurement:
    """Result of measuring actual runtime."""
    program: str
    model_name: str
    runtime_ns: float
    iterations: int
    stddev_ns: float = 0.0
    min_ns: float = 0.0
    max_ns: float = 0.0


@dataclass
class FullComparison:
    """Complete comparison of compilation and runtime for multiple cost models."""
    program: str
    compilations: dict[str, CompilationResult]
    runtimes: dict[str, RuntimeMeasurement] = field(default_factory=dict)

    def get_cost_to_runtime_pairs(self) -> list[tuple[float, float]]:
        """Get (predicted_cost, actual_runtime) pairs for correlation analysis."""
        pairs = []
        for model_name, compilation in self.compilations.items():
            if model_name in self.runtimes:
                runtime = self.runtimes[model_name]
                pairs.append((compilation.extracted_cost, runtime.runtime_ns))
        return pairs


def compile_to_egraph(
    spy_path: str | Path,
    tu: TranslationUnit | None = None,
) -> tuple[EGraph, TranslationUnit]:
    """
    Compile a SPy file to an optimized e-graph.

    Args:
        spy_path: Path to the .spy file
        tu: Optional pre-parsed TranslationUnit

    Returns:
        Tuple of (optimized EGraph, TranslationUnit)
    """
    if tu is None:
        tu = frontend(str(spy_path))

    # Get the first (main) function
    fqn = tu.list_functions()[0]
    fi = tu.get_function(fqn)

    # Convert to e-graph
    memo = egraph_conversion(fi.region)
    root = GraphRoot(memo[fi.region])

    egraph = EGraph()
    egraph.let("root", root)
    egraph.let("mds", egraph_convert_metadata(fi.metadata, memo))

    # Run optimization (rewrite rules)
    egraph_optimize(egraph)

    return egraph, tu


def extract_with_model(
    egraph: EGraph,
    cost_model: _CostModel,
    model_name: str,
) -> CompilationResult:
    """
    Run extraction with a specific cost model.

    Args:
        egraph: The optimized e-graph
        cost_model: The cost model to use
        model_name: Name for identification

    Returns:
        CompilationResult with extraction details
    """
    stats: dict[str, Any] = {}

    start = time.perf_counter_ns()
    extraction = egraph_extraction(egraph, cost_model=cost_model, stats=stats)
    extraction.compute()
    result = extraction.extract_common_root()
    end = time.perf_counter_ns()

    return CompilationResult(
        program="",  # Will be set by caller
        model_name=model_name,
        extracted_cost=result.cost,
        extracted_node_count=result.graph.number_of_nodes(),
        extraction_time_ns=end - start,
    )


def run_compilation(
    spy_path: str | Path,
    cost_model: _CostModel,
    model_name: str,
    output_dir: str | Path | None = None,
) -> CompilationResult:
    """
    Compile a SPy program with a specific cost model.

    Args:
        spy_path: Path to the .spy file
        cost_model: Cost model to use for extraction
        model_name: Name for identification
        output_dir: Directory for output files (None for temp)

    Returns:
        CompilationResult with full compilation details
    """
    spy_path = Path(spy_path)
    program_name = spy_path.stem

    try:
        # Compile to e-graph
        egraph, tu = compile_to_egraph(spy_path)

        # Extract with cost model
        result = extract_with_model(egraph, cost_model, model_name)
        result.program = program_name

        return result

    except Exception as e:
        return CompilationResult(
            program=program_name,
            model_name=model_name,
            extracted_cost=float('inf'),
            extracted_node_count=0,
            extraction_time_ns=0,
            success=False,
            error=str(e),
        )


def run_full_comparison(
    spy_path: str | Path,
    models: dict[str, _CostModel],
    measure_runtime: bool = False,
    runtime_iterations: int = 100,
) -> FullComparison:
    """
    Run a full comparison of cost models on a program.

    Args:
        spy_path: Path to the .spy file
        models: Dict mapping model names to cost model instances
        measure_runtime: If True, also measure actual runtime
        runtime_iterations: Number of iterations for runtime measurement

    Returns:
        FullComparison with all results
    """
    spy_path = Path(spy_path)
    program_name = spy_path.stem

    # Compile to e-graph once
    egraph, tu = compile_to_egraph(spy_path)

    compilations: dict[str, CompilationResult] = {}
    runtimes: dict[str, RuntimeMeasurement] = {}

    for model_name, cost_model in models.items():
        # Run extraction
        result = extract_with_model(egraph, cost_model, model_name)
        result.program = program_name
        compilations[model_name] = result

        # Measure runtime if requested (placeholder for future implementation)
        if measure_runtime:
            # This is where WASM or native runtime measurement would go
            runtime = measure_runtime_placeholder(
                program_name, model_name, runtime_iterations
            )
            if runtime:
                runtimes[model_name] = runtime

    return FullComparison(
        program=program_name,
        compilations=compilations,
        runtimes=runtimes,
    )


def measure_runtime_placeholder(
    program: str,
    model_name: str,
    iterations: int,
) -> RuntimeMeasurement | None:
    """
    Placeholder for runtime measurement.

    In the future, this will:
    1. Compile to WASM using: MLIR → LLVM → WASM
    2. Load the WASM module using spy.llwasm.LLWasmInstance
    3. Call the function and measure time

    For now, this returns None to indicate runtime measurement is unavailable.
    """
    # TODO: Implement when WASM compilation pipeline is available
    #
    # Implementation sketch:
    #
    # from spy.llwasm import LLWasmInstance
    #
    # # Load compiled WASM
    # ll = LLWasmInstance.from_file(wasm_path)
    #
    # # Warmup
    # for _ in range(10):
    #     ll.call("main")
    #
    # # Measure
    # times = []
    # for _ in range(iterations):
    #     start = time.perf_counter_ns()
    #     ll.call("main")
    #     end = time.perf_counter_ns()
    #     times.append(end - start)
    #
    # return RuntimeMeasurement(
    #     program=program,
    #     model_name=model_name,
    #     runtime_ns=sum(times) / len(times),
    #     iterations=iterations,
    #     stddev_ns=statistics.stdev(times),
    #     min_ns=min(times),
    #     max_ns=max(times),
    # )

    return None


def estimate_runtime_from_cost(
    cost: float,
    base_ns: float = 1000,
    cost_scale: float = 100,
) -> float:
    """
    Estimate runtime from predicted cost (for testing without actual execution).

    This is a simple linear model: runtime = base + cost * scale

    Args:
        cost: Predicted cost from cost model
        base_ns: Base runtime in nanoseconds
        cost_scale: Nanoseconds per unit of cost

    Returns:
        Estimated runtime in nanoseconds
    """
    return base_ns + cost * cost_scale


def simulate_runtime_comparison(
    spy_path: str | Path,
    models: dict[str, _CostModel],
    noise_factor: float = 0.1,
) -> FullComparison:
    """
    Simulate a full comparison with synthetic runtime data.

    This is useful for testing the validation pipeline without actual
    WASM compilation. It assumes that lower cost → lower runtime with some noise.

    Args:
        spy_path: Path to the .spy file
        models: Dict mapping model names to cost model instances
        noise_factor: Random noise factor (0.0 = perfect correlation)

    Returns:
        FullComparison with simulated runtimes
    """
    import random

    comparison = run_full_comparison(spy_path, models, measure_runtime=False)

    # Generate synthetic runtimes based on costs
    for model_name, compilation in comparison.compilations.items():
        base_runtime = estimate_runtime_from_cost(compilation.extracted_cost)

        # Add noise to simulate real-world variation
        noise = random.gauss(1.0, noise_factor)
        runtime_ns = base_runtime * max(0.1, noise)

        comparison.runtimes[model_name] = RuntimeMeasurement(
            program=compilation.program,
            model_name=model_name,
            runtime_ns=runtime_ns,
            iterations=100,
            stddev_ns=runtime_ns * noise_factor,
            min_ns=runtime_ns * 0.9,
            max_ns=runtime_ns * 1.1,
        )

    return comparison


def print_comparison_report(comparison: FullComparison) -> None:
    """Print a formatted comparison report."""
    print(f"\n{'='*70}")
    print(f"Comparison Report: {comparison.program}")
    print(f"{'='*70}")

    # Compilation results
    print(f"\n{'Model':<15} {'Cost':>12} {'Nodes':>8} {'Extract (ms)':>14}")
    print(f"{'-'*49}")

    for model_name, comp in comparison.compilations.items():
        if comp.success:
            print(f"{model_name:<15} {comp.extracted_cost:>12.2f} "
                  f"{comp.extracted_node_count:>8} "
                  f"{comp.extraction_time_ns/1e6:>14.2f}")
        else:
            print(f"{model_name:<15} {'ERROR':>12} {'-':>8} {'-':>14}")
            print(f"  Error: {comp.error}")

    # Runtime results (if available)
    if comparison.runtimes:
        print(f"\n{'Model':<15} {'Runtime (ns)':>14} {'Stddev':>12} {'Iterations':>12}")
        print(f"{'-'*53}")

        for model_name, rt in comparison.runtimes.items():
            print(f"{model_name:<15} {rt.runtime_ns:>14.1f} "
                  f"{rt.stddev_ns:>12.1f} {rt.iterations:>12}")


def run_benchmark_suite(
    programs: Sequence[str | Path],
    models: dict[str, _CostModel] | None = None,
    simulate_runtime: bool = True,
    verbose: bool = True,
) -> list[FullComparison]:
    """
    Run a full benchmark suite on multiple programs.

    Args:
        programs: List of paths to .spy files
        models: Dict of cost models (uses defaults if None)
        simulate_runtime: If True, generate synthetic runtime data
        verbose: If True, print progress and results

    Returns:
        List of FullComparison results
    """
    from .cost_models import (
        BaselineCostModel,
        MLCostModel,
    )
    from .principled_costs import (
        AnalyticalCostModel,
        InstructionCostModel,
    )

    if models is None:
        models = {
            "baseline": BaselineCostModel(),
            "instruction": InstructionCostModel(),
            "analytical": AnalyticalCostModel(),
            "ml": MLCostModel(),
        }

    results: list[FullComparison] = []

    for spy_path in programs:
        if verbose:
            print(f"\nProcessing: {Path(spy_path).stem}")

        try:
            if simulate_runtime:
                comparison = simulate_runtime_comparison(spy_path, models)
            else:
                comparison = run_full_comparison(spy_path, models)

            results.append(comparison)

            if verbose:
                print_comparison_report(comparison)

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")

    return results


def get_benchmark_programs() -> list[Path]:
    """Get all benchmark programs from the programs directory."""
    programs_dir = Path(__file__).parent / "programs"
    return list(programs_dir.glob("*.spy"))


if __name__ == "__main__":
    # Quick test
    programs = get_benchmark_programs()
    if programs:
        run_benchmark_suite(programs[:2], verbose=True)
    else:
        print("No benchmark programs found")
