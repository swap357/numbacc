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

    # Analyze extraction differences
    from nbcc.experiments import analyze_extraction, compare_models
    analysis = analyze_extraction(egraph, cost_model)
    print_extraction_report(analysis)

    # Validate cost model accuracy
    from nbcc.experiments import validate_cost_model, print_validation_report
    result = validate_cost_model(predictions, actuals, "my_model")
    print_validation_report({"my_model": result})
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
from .extraction_analysis import (
    EClassAlternatives,
    EClassDisagreement,
    ExtractionAnalysis,
    ModelComparison,
    analyze_extraction,
    compare_models,
    find_disagreements,
    get_extraction_summary,
    print_comparison_report,
    print_extraction_report,
)
from .constant_folding import (
    create_constant_folding_schedule,
    get_constant_folding_rulesets,
    make_constant_folding_schedule,
    ruleset_constant_fold,
    ruleset_constant_fold_division,
    ruleset_constant_fold_unary,
)
from .runtime_measurement import (
    CompilationResult,
    FullComparison,
    RuntimeMeasurement,
    compile_to_egraph,
    extract_with_model,
    get_benchmark_programs,
    print_comparison_report as print_runtime_comparison_report,
    run_benchmark_suite,
    run_compilation,
    run_full_comparison,
    simulate_runtime_comparison,
)
from .validation_metrics import (
    DetailedValidation,
    PairwiseComparison,
    ValidationResult,
    compare_cost_model_quality,
    compute_accuracy,
    compute_kendall_tau,
    compute_mae,
    compute_rmse,
    compute_spearman,
    print_detailed_validation_report,
    print_validation_report,
    validate_cost_model,
    validate_cost_model_detailed,
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
    # Extraction Analysis (Phase 1)
    "EClassAlternatives",
    "EClassDisagreement",
    "ExtractionAnalysis",
    "ModelComparison",
    "analyze_extraction",
    "compare_models",
    "find_disagreements",
    "get_extraction_summary",
    "print_extraction_report",
    "print_comparison_report",
    # Constant Folding (Phase 2)
    "ruleset_constant_fold",
    "ruleset_constant_fold_unary",
    "ruleset_constant_fold_division",
    "get_constant_folding_rulesets",
    "create_constant_folding_schedule",
    "make_constant_folding_schedule",
    # Runtime Measurement (Phase 4)
    "CompilationResult",
    "RuntimeMeasurement",
    "FullComparison",
    "compile_to_egraph",
    "extract_with_model",
    "run_compilation",
    "run_full_comparison",
    "simulate_runtime_comparison",
    "run_benchmark_suite",
    "get_benchmark_programs",
    "print_runtime_comparison_report",
    # Validation Metrics (Phase 5)
    "ValidationResult",
    "PairwiseComparison",
    "DetailedValidation",
    "compute_spearman",
    "compute_kendall_tau",
    "compute_accuracy",
    "compute_mae",
    "compute_rmse",
    "validate_cost_model",
    "validate_cost_model_detailed",
    "print_validation_report",
    "print_detailed_validation_report",
    "compare_cost_model_quality",
]
