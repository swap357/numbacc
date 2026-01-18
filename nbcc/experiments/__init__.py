"""
Cost Model Experiments for numbacc.

This package provides tools for comparing different cost model strategies
for e-graph extraction. The goal is to find cost models that better predict
actual runtime, leading to better extraction decisions and faster generated code.

Cost Models:
- BaselineCostModel: Static costs per operation type (current approach)
- InstructionCostModel: Principled costs based on CPU instruction latencies
- CalibratedCostModel: Calibration-ready model for real measurements
- MLCostModel/TrainedMLCostModel: ML-based predictions from node features
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

    # Principled cost model based on instruction latencies
    from nbcc.experiments import InstructionCostModel
    model = InstructionCostModel()

    # Train ML model on synthetic data
    from nbcc.experiments import create_pretrained_model
    predictor = create_pretrained_model()
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
from .principled_costs import (
    AnalyticalCostModel,
    CalibratedCostModel,
    InstructionCost,
    InstructionCostModel,
    OPERATION_COSTS,
    create_cost_comparison_table,
)
from .synthetic_data import (
    OperationMix,
    SyntheticDataGenerator,
    SyntheticProgram,
    generate_training_data,
)
from .ml_training import (
    CostPredictor,
    TrainedMLCostModel,
    TrainingDataset,
    TrainingSample,
    create_pretrained_model,
    evaluate_model,
    generate_training_data_from_operations,
    train_cost_model,
)

__all__ = [
    # Cost Models (original)
    "BaselineCostModel",
    "PerfCostModel",
    "MLCostModel",
    "HybridCostModel",
    "get_cost_model",
    "CostModelStats",
    # Cost Models (principled)
    "InstructionCostModel",
    "AnalyticalCostModel",
    "CalibratedCostModel",
    "InstructionCost",
    "OPERATION_COSTS",
    "create_cost_comparison_table",
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
    # Synthetic Data
    "SyntheticProgram",
    "SyntheticDataGenerator",
    "OperationMix",
    "generate_training_data",
    # ML Training
    "CostPredictor",
    "TrainedMLCostModel",
    "TrainingDataset",
    "TrainingSample",
    "train_cost_model",
    "evaluate_model",
    "create_pretrained_model",
    "generate_training_data_from_operations",
]
