"""
Benchmark runner for cost model experiments.

Compares different cost models by measuring their correlation with actual
runtime performance. Uses Spearman correlation as the primary quality metric.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from egglog import EGraph
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import CostModel as _CostModel
from sealir.eqsat.rvsdg_extract import egraph_extraction

from nbcc.egraph.rules import egraph_convert_metadata, egraph_optimize
from nbcc.frontend import TranslationUnit, frontend

from .cost_models import (
    BaselineCostModel,
    HybridCostModel,
    MLCostModel,
    PerfCostModel,
    get_cost_model,
)


@dataclass
class ExtractionResult:
    """Result of running extraction with a cost model."""
    program: str
    model_name: str
    predicted_cost: float
    node_count: int
    extraction_time_ns: int
    extraction_iterations: int


@dataclass
class RuntimeResult:
    """Result of measuring actual runtime."""
    program: str
    runtime_ns: float
    iterations: int


@dataclass
class ComparisonResult:
    """Full comparison result for a program across models."""
    program: str
    results: dict[str, ExtractionResult]
    actual_runtime_ns: float | None = None


def compile_to_egraph(
    spy_path: str | Path,
    tu: TranslationUnit | None = None,
) -> tuple[EGraph, TranslationUnit]:
    """
    Compile a SPy file to an e-graph (without extraction).

    Args:
        spy_path: Path to the .spy file
        tu: Optional pre-parsed TranslationUnit

    Returns:
        Tuple of (EGraph, TranslationUnit)
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


def run_extraction(
    egraph: EGraph,
    cost_model: _CostModel,
) -> tuple[float, int, int, int]:
    """
    Run extraction with a given cost model.

    Args:
        egraph: The e-graph to extract from
        cost_model: The cost model to use

    Returns:
        Tuple of (predicted_cost, node_count, time_ns, iterations)
    """
    stats: dict[str, Any] = {}

    start = time.perf_counter_ns()
    extraction = egraph_extraction(egraph, cost_model=cost_model, stats=stats)
    extraction.compute()
    result = extraction.extract_common_root()
    end = time.perf_counter_ns()

    # Count nodes in extracted graph
    node_count = result.graph.number_of_nodes()

    iterations = stats.get("extraction_iteration_count", 0)

    return result.cost, node_count, end - start, iterations


def count_nodes_in_result(result: Any) -> int:
    """Count nodes in an extraction result."""
    if hasattr(result, "graph"):
        return result.graph.number_of_nodes()
    return 0


def compare_cost_models(
    programs: list[str | Path],
    models: dict[str, _CostModel] | None = None,
    verbose: bool = False,
) -> list[ComparisonResult]:
    """
    Compare different cost models on a set of programs.

    Args:
        programs: List of paths to .spy files
        models: Dict mapping model names to CostModel instances.
                If None, uses default set of all 4 models.
        verbose: If True, print progress

    Returns:
        List of ComparisonResult for each program
    """
    if models is None:
        models = {
            "baseline": BaselineCostModel(),
            "perf": PerfCostModel(calibrate=True),
            "xgboost": MLCostModel(),
            "hybrid": HybridCostModel(),
        }

    results: list[ComparisonResult] = []

    for prog_path in programs:
        prog_name = Path(prog_path).stem
        if verbose:
            print(f"Processing: {prog_name}")

        # Compile to e-graph once
        try:
            egraph, tu = compile_to_egraph(prog_path)
        except Exception as e:
            if verbose:
                print(f"  Error compiling {prog_name}: {e}")
            continue

        # Run extraction with each model
        model_results: dict[str, ExtractionResult] = {}
        for model_name, model in models.items():
            try:
                cost, nodes, time_ns, iters = run_extraction(egraph, model)
                model_results[model_name] = ExtractionResult(
                    program=prog_name,
                    model_name=model_name,
                    predicted_cost=cost,
                    node_count=nodes,
                    extraction_time_ns=time_ns,
                    extraction_iterations=iters,
                )
                if verbose:
                    print(f"  {model_name}: cost={cost:.2f}, nodes={nodes}")
            except Exception as e:
                if verbose:
                    print(f"  Error with {model_name}: {e}")

        results.append(ComparisonResult(
            program=prog_name,
            results=model_results,
        ))

    return results


def compute_spearman_correlation(
    predicted: list[float],
    actual: list[float],
) -> float:
    """
    Compute Spearman rank correlation between predicted and actual costs.

    Args:
        predicted: List of predicted costs
        actual: List of actual runtimes

    Returns:
        Spearman correlation coefficient (-1 to 1)
        1.0 = perfect ranking (extraction always picks fastest)
        0.0 = random (cost model is useless)
        -1.0 = perfectly wrong (always picks slowest)
    """
    if len(predicted) != len(actual) or len(predicted) < 2:
        return 0.0

    try:
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(predicted, actual)
        return correlation if correlation == correlation else 0.0  # Handle NaN
    except ImportError:
        # Fallback: simple rank correlation
        return _simple_rank_correlation(predicted, actual)


def _simple_rank_correlation(x: list[float], y: list[float]) -> float:
    """Simple Spearman correlation without scipy."""
    n = len(x)
    if n < 2:
        return 0.0

    # Compute ranks
    def rank(values: list[float]) -> list[float]:
        sorted_idx = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_idx):
            ranks[idx] = float(rank_val)
        return ranks

    rx = rank(x)
    ry = rank(y)

    # Compute Spearman correlation
    d_squared = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - (6 * d_squared) / (n * (n * n - 1))


def analyze_results(
    results: list[ComparisonResult],
    actual_runtimes: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Analyze comparison results and compute correlations.

    Args:
        results: List of ComparisonResult from compare_cost_models
        actual_runtimes: Optional dict mapping program names to actual runtimes

    Returns:
        Dict mapping model names to their metrics
    """
    # Get all model names
    model_names: set[str] = set()
    for r in results:
        model_names.update(r.results.keys())

    # Collect data per model
    model_data: dict[str, dict[str, Any]] = {
        name: {
            "costs": [],
            "node_counts": [],
            "times_ns": [],
            "programs": [],
        }
        for name in model_names
    }

    for r in results:
        for model_name, ext_result in r.results.items():
            model_data[model_name]["costs"].append(ext_result.predicted_cost)
            model_data[model_name]["node_counts"].append(ext_result.node_count)
            model_data[model_name]["times_ns"].append(ext_result.extraction_time_ns)
            model_data[model_name]["programs"].append(r.program)

    # Compute metrics
    metrics: dict[str, dict[str, float]] = {}
    for model_name, data in model_data.items():
        costs = data["costs"]
        node_counts = data["node_counts"]
        times = data["times_ns"]

        metrics[model_name] = {
            "avg_cost": sum(costs) / len(costs) if costs else 0,
            "avg_node_count": sum(node_counts) / len(node_counts) if node_counts else 0,
            "avg_extraction_time_ms": sum(times) / len(times) / 1e6 if times else 0,
        }

        # Compute correlation with actual runtime if available
        if actual_runtimes:
            programs = data["programs"]
            actual = [actual_runtimes.get(p, 0) for p in programs]
            if any(a > 0 for a in actual):
                correlation = compute_spearman_correlation(costs, actual)
                metrics[model_name]["spearman_correlation"] = correlation

    return metrics


def print_comparison_table(
    results: list[ComparisonResult],
    metrics: dict[str, dict[str, float]] | None = None,
) -> None:
    """Print a formatted comparison table."""
    if not results:
        print("No results to display")
        return

    # Get model names
    model_names = sorted(set(
        name for r in results for name in r.results.keys()
    ))

    # Header
    print("\n" + "=" * 80)
    print("Cost Model Comparison Results")
    print("=" * 80)

    # Per-program results
    print(f"\n{'Program':<15}", end="")
    for name in model_names:
        print(f"{name:>12}", end="")
    print()
    print("-" * (15 + 12 * len(model_names)))

    for r in results:
        print(f"{r.program:<15}", end="")
        for name in model_names:
            if name in r.results:
                cost = r.results[name].predicted_cost
                print(f"{cost:>12.2f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

    # Summary metrics
    if metrics:
        print("\n" + "-" * 80)
        print("Summary Metrics")
        print("-" * 80)

        print(f"\n{'Metric':<25}", end="")
        for name in model_names:
            print(f"{name:>12}", end="")
        print()
        print("-" * (25 + 12 * len(model_names)))

        # Avg cost
        print(f"{'Avg Predicted Cost':<25}", end="")
        for name in model_names:
            if name in metrics:
                print(f"{metrics[name]['avg_cost']:>12.2f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

        # Avg node count
        print(f"{'Avg Node Count':<25}", end="")
        for name in model_names:
            if name in metrics:
                print(f"{metrics[name]['avg_node_count']:>12.1f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

        # Avg extraction time
        print(f"{'Avg Extraction Time (ms)':<25}", end="")
        for name in model_names:
            if name in metrics:
                print(f"{metrics[name]['avg_extraction_time_ms']:>12.2f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

        # Correlation (if available)
        if any("spearman_correlation" in metrics.get(n, {}) for n in model_names):
            print(f"{'Spearman Correlation':<25}", end="")
            for name in model_names:
                if name in metrics and "spearman_correlation" in metrics[name]:
                    corr = metrics[name]["spearman_correlation"]
                    print(f"{corr:>12.3f}", end="")
                else:
                    print(f"{'N/A':>12}", end="")
            print()

    print("\n" + "=" * 80)


@dataclass
class BenchmarkRunner:
    """
    Main benchmark runner for cost model experiments.

    Usage:
        runner = BenchmarkRunner()
        runner.add_program("arithmetic.spy")
        runner.add_program("branches.spy")
        runner.add_program("loops.spy")

        results = runner.run()
        runner.print_results()
    """

    programs: list[Path] = field(default_factory=list)
    models: dict[str, _CostModel] = field(default_factory=dict)
    results: list[ComparisonResult] = field(default_factory=list)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    actual_runtimes: dict[str, float] = field(default_factory=dict)
    verbose: bool = True

    def __post_init__(self) -> None:
        if not self.models:
            self.models = {
                "baseline": BaselineCostModel(),
                "perf": PerfCostModel(calibrate=False),  # Calibrate on run
                "xgboost": MLCostModel(),
                "hybrid": HybridCostModel(),
            }

    def add_program(self, path: str | Path) -> None:
        """Add a program to the benchmark."""
        self.programs.append(Path(path))

    def add_programs_from_dir(self, dir_path: str | Path, pattern: str = "*.spy") -> None:
        """Add all matching programs from a directory."""
        dir_path = Path(dir_path)
        for p in dir_path.glob(pattern):
            self.programs.append(p)

    def run(self, calibrate_perf: bool = True) -> list[ComparisonResult]:
        """
        Run the benchmark.

        Args:
            calibrate_perf: If True, calibrate PerfCostModel before running

        Returns:
            List of ComparisonResult
        """
        if calibrate_perf:
            if "perf" in self.models and isinstance(self.models["perf"], PerfCostModel):
                if self.verbose:
                    print("Calibrating PerfCostModel...")
                self.models["perf"]._calibrate()

        if self.verbose:
            print(f"\nRunning benchmark on {len(self.programs)} programs...")

        self.results = compare_cost_models(
            self.programs,
            self.models,
            verbose=self.verbose,
        )

        self.metrics = analyze_results(self.results, self.actual_runtimes)

        return self.results

    def print_results(self) -> None:
        """Print the benchmark results."""
        print_comparison_table(self.results, self.metrics)

    def get_model_ranking(self) -> list[tuple[str, float]]:
        """
        Get models ranked by their performance.

        Returns list of (model_name, score) tuples, sorted by score.
        Score is Spearman correlation if available, else inverse of avg_cost.
        """
        rankings = []
        for name, m in self.metrics.items():
            if "spearman_correlation" in m:
                score = m["spearman_correlation"]
            else:
                # Use inverse of avg_cost as proxy
                score = 1.0 / (m["avg_cost"] + 1)
            rankings.append((name, score))

        return sorted(rankings, key=lambda x: -x[1])


def get_test_programs_dir() -> Path:
    """Get the path to the test programs directory."""
    return Path(__file__).parent / "programs"


def run_default_benchmark(verbose: bool = True) -> BenchmarkRunner:
    """
    Run the default benchmark with all test programs.

    Returns:
        BenchmarkRunner with results
    """
    runner = BenchmarkRunner(verbose=verbose)

    # Add test programs
    programs_dir = get_test_programs_dir()
    if programs_dir.exists():
        runner.add_programs_from_dir(programs_dir)
    else:
        if verbose:
            print(f"Warning: Test programs directory not found: {programs_dir}")

    if not runner.programs:
        if verbose:
            print("No programs to benchmark. Add .spy files to the programs/ directory.")
        return runner

    runner.run()
    runner.print_results()

    return runner


if __name__ == "__main__":
    run_default_benchmark()
