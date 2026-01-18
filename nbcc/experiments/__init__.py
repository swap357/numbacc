"""
Cost Model Experiments for numbacc.

This package provides tools for comparing different cost model strategies
for e-graph extraction. The goal is to find cost models that better predict
actual runtime, leading to better extraction decisions and faster generated code.

Cost Models:
- BaselineCostModel: Static costs per operation type (current approach)
- PerfCostModel: Hardware counter-based measurements
- MLCostModel: XGBoost-based predictions from node features
- HybridCostModel: Weighted combination of Perf and ML models

Quality Metric:
- Spearman correlation between predicted cost and actual runtime
- Correlation = 1.0: Perfect ranking (extraction always picks fastest)
- Correlation = 0.0: Random (cost model is useless)

Usage:
    from nbcc.experiments import BenchmarkRunner, get_cost_model

    # Quick start: run default benchmark
    runner = BenchmarkRunner()
    runner.add_programs_from_dir("path/to/spy/files")
    runner.run()
    runner.print_results()

    # Custom cost model
    from nbcc.experiments import BaselineCostModel, MLCostModel
    baseline = BaselineCostModel()
    ml_model = MLCostModel()

    # Compare on specific programs
    from nbcc.experiments import compare_cost_models
    results = compare_cost_models(
        ["program1.spy", "program2.spy"],
        models={"baseline": baseline, "ml": ml_model},
    )
"""

from .benchmark import (
    BenchmarkRunner,
    ComparisonResult,
    ExtractionResult,
    analyze_results,
    compare_cost_models,
    compute_spearman_correlation,
    get_test_programs_dir,
    print_comparison_table,
    run_default_benchmark,
)
from .cost_models import (
    BaselineCostModel,
    CostModelStats,
    HybridCostModel,
    MLCostModel,
    PerfCostModel,
    get_cost_model,
)
from .features import (
    ExtendedNodeFeatures,
    FeatureCollector,
    NodeFeatures,
    OpCategory,
    extract_extended_features,
    extract_features,
)

__all__ = [
    # Cost Models
    "BaselineCostModel",
    "PerfCostModel",
    "MLCostModel",
    "HybridCostModel",
    "get_cost_model",
    "CostModelStats",
    # Features
    "NodeFeatures",
    "ExtendedNodeFeatures",
    "OpCategory",
    "extract_features",
    "extract_extended_features",
    "FeatureCollector",
    # Benchmark
    "BenchmarkRunner",
    "ExtractionResult",
    "ComparisonResult",
    "compare_cost_models",
    "analyze_results",
    "print_comparison_table",
    "compute_spearman_correlation",
    "get_test_programs_dir",
    "run_default_benchmark",
]
