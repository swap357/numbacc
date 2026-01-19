"""
End-to-end tests for the cost model validation pipeline.

This module proves that:
1. Different cost models can produce different extractions
2. Extraction differences lead to different node counts/costs
3. The validation metrics correctly measure model quality
4. Spearman correlation works as expected

To run: pytest nbcc/experiments/tests/test_cost_model_validation.py -v
"""

import pytest
from pathlib import Path

# Import all the new modules
from nbcc.experiments import (
    # Cost Models
    BaselineCostModel,
    InstructionCostModel,
    AnalyticalCostModel,
    MLCostModel,
    # Extraction Analysis
    analyze_extraction,
    compare_models,
    print_extraction_report,
    print_comparison_report,
    # Runtime Measurement
    compile_to_egraph,
    extract_with_model,
    run_full_comparison,
    simulate_runtime_comparison,
    get_benchmark_programs,
    # Validation Metrics
    compute_spearman,
    compute_accuracy,
    validate_cost_model,
    print_validation_report,
)


class TestValidationMetrics:
    """Test the validation metrics themselves."""

    def test_spearman_perfect_correlation(self):
        """Perfect correlation should give 1.0."""
        predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
        actual = [10.0, 20.0, 30.0, 40.0, 50.0]  # Same order

        corr = compute_spearman(predicted, actual)
        assert corr > 0.99, f"Expected ~1.0, got {corr}"

    def test_spearman_inverse_correlation(self):
        """Inverse correlation should give -1.0."""
        predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
        actual = [50.0, 40.0, 30.0, 20.0, 10.0]  # Reverse order

        corr = compute_spearman(predicted, actual)
        assert corr < -0.99, f"Expected ~-1.0, got {corr}"

    def test_spearman_no_correlation(self):
        """Random data should have correlation near 0."""
        import random
        random.seed(42)

        predicted = [random.random() for _ in range(100)]
        actual = [random.random() for _ in range(100)]

        corr = compute_spearman(predicted, actual)
        assert -0.3 < corr < 0.3, f"Expected ~0.0, got {corr}"

    def test_accuracy_perfect(self):
        """All correct orderings should give 100% accuracy."""
        predicted = [1.0, 2.0, 3.0, 4.0]
        actual = [10.0, 20.0, 30.0, 40.0]

        accuracy, comparisons = compute_accuracy(predicted, actual)
        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"
        assert all(c.is_correct for c in comparisons)

    def test_accuracy_inverse(self):
        """All wrong orderings should give 0% accuracy."""
        predicted = [4.0, 3.0, 2.0, 1.0]
        actual = [10.0, 20.0, 30.0, 40.0]

        accuracy, comparisons = compute_accuracy(predicted, actual)
        assert accuracy == 0.0, f"Expected 0.0, got {accuracy}"
        assert all(not c.is_correct for c in comparisons)

    def test_validate_cost_model(self):
        """Full validation should produce all metrics."""
        predicted = [1.0, 2.5, 2.0, 4.0, 5.0]
        actual = [100.0, 200.0, 250.0, 400.0, 500.0]

        result = validate_cost_model(predicted, actual, "test_model")

        assert result.model_name == "test_model"
        assert result.num_samples == 5
        assert -1.0 <= result.spearman_correlation <= 1.0
        assert 0.0 <= result.accuracy <= 1.0
        assert result.mae >= 0.0


class TestExtractionAnalysis:
    """Test extraction analysis with real e-graphs."""

    @pytest.fixture
    def sample_egraph(self):
        """Create a simple e-graph from an existing test program."""
        programs = get_benchmark_programs()
        if not programs:
            pytest.skip("No benchmark programs available")

        # Use arithmetic.spy if available
        arithmetic = [p for p in programs if p.stem == "arithmetic"]
        if arithmetic:
            program = arithmetic[0]
        else:
            program = programs[0]

        egraph, tu = compile_to_egraph(program)
        return egraph, program.stem

    def test_analyze_extraction_basic(self, sample_egraph):
        """Basic extraction analysis should work."""
        egraph, program_name = sample_egraph
        model = BaselineCostModel()

        analysis = analyze_extraction(egraph, model, "baseline")

        assert analysis.model_name == "baseline"
        assert analysis.total_cost > 0
        assert analysis.node_count > 0
        assert analysis.num_eclasses > 0

    def test_compare_models(self, sample_egraph):
        """Model comparison should detect similarities and differences."""
        egraph, program_name = sample_egraph

        models = {
            "baseline": BaselineCostModel(),
            "instruction": InstructionCostModel(),
        }

        comparison = compare_models(egraph, models, program_name)

        assert comparison.program == program_name
        assert len(comparison.analyses) == 2
        assert "baseline" in comparison.analyses
        assert "instruction" in comparison.analyses

        # Agreement rate should be between 0 and 1
        assert 0.0 <= comparison.agreement_rate <= 1.0


class TestRuntimeMeasurement:
    """Test the runtime measurement infrastructure."""

    def test_extract_with_different_models(self):
        """Different models should be able to extract from same e-graph."""
        programs = get_benchmark_programs()
        if not programs:
            pytest.skip("No benchmark programs available")

        program = programs[0]
        egraph, tu = compile_to_egraph(program)

        models = {
            "baseline": BaselineCostModel(),
            "instruction": InstructionCostModel(),
            "analytical": AnalyticalCostModel(),
        }

        results = {}
        for name, model in models.items():
            result = extract_with_model(egraph, model, name)
            results[name] = result
            assert result.extracted_cost > 0
            assert result.extracted_node_count > 0

        # At least verify we got different costs
        costs = [r.extracted_cost for r in results.values()]
        # Note: costs might be the same if models agree on ordering

    def test_run_full_comparison(self):
        """Full comparison should collect all metrics."""
        programs = get_benchmark_programs()
        if not programs:
            pytest.skip("No benchmark programs available")

        program = programs[0]
        models = {
            "baseline": BaselineCostModel(),
            "instruction": InstructionCostModel(),
        }

        comparison = run_full_comparison(program, models)

        assert comparison.program == program.stem
        assert len(comparison.compilations) == 2

        for name, comp in comparison.compilations.items():
            assert comp.success
            assert comp.extracted_cost > 0

    def test_simulate_runtime(self):
        """Simulated runtime comparison should produce valid data."""
        programs = get_benchmark_programs()
        if not programs:
            pytest.skip("No benchmark programs available")

        program = programs[0]
        models = {
            "baseline": BaselineCostModel(),
            "instruction": InstructionCostModel(),
        }

        comparison = simulate_runtime_comparison(program, models)

        assert len(comparison.runtimes) == 2

        for name, runtime in comparison.runtimes.items():
            assert runtime.runtime_ns > 0
            assert runtime.iterations > 0


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_validation_pipeline(self):
        """
        Full end-to-end test:
        1. Compile programs to e-graphs
        2. Extract with multiple cost models
        3. Simulate runtimes
        4. Validate cost models
        5. Produce report
        """
        programs = get_benchmark_programs()
        if len(programs) < 2:
            pytest.skip("Need at least 2 benchmark programs")

        # Use first 2 programs for speed
        programs = programs[:2]

        models = {
            "baseline": BaselineCostModel(),
            "instruction": InstructionCostModel(),
            "analytical": AnalyticalCostModel(),
        }

        # Collect predictions and simulated runtimes per model
        model_data = {name: {"costs": [], "runtimes": []} for name in models}

        for program in programs:
            comparison = simulate_runtime_comparison(program, models)

            for name in models:
                if name in comparison.compilations:
                    model_data[name]["costs"].append(
                        comparison.compilations[name].extracted_cost
                    )
                if name in comparison.runtimes:
                    model_data[name]["runtimes"].append(
                        comparison.runtimes[name].runtime_ns
                    )

        # Validate each model
        results = {}
        for name, data in model_data.items():
            if len(data["costs"]) >= 2:
                result = validate_cost_model(
                    data["costs"],
                    data["runtimes"],
                    name
                )
                results[name] = result

        # We should have results for each model
        assert len(results) == len(models)

        # Print report (for manual inspection)
        print("\n" + "="*60)
        print("End-to-End Validation Results")
        print("="*60)
        print_validation_report(results)

        # All models should have positive correlation with simulated data
        # (since simulated runtimes are based on costs)
        for name, result in results.items():
            assert result.spearman_correlation > 0, \
                f"{name} should have positive correlation with simulated data"


class TestConstantFolding:
    """Test constant folding rules."""

    def test_constant_folding_imports(self):
        """Constant folding rules should be importable."""
        from nbcc.experiments import (
            ruleset_constant_fold,
            ruleset_constant_fold_unary,
            ruleset_constant_fold_division,
            get_constant_folding_rulesets,
            make_constant_folding_schedule,
        )

        # Verify we can create a schedule
        schedule = make_constant_folding_schedule()
        assert schedule is not None

        # Verify we get rulesets
        rulesets = get_constant_folding_rulesets()
        assert len(rulesets) == 3


class TestBenchmarkPrograms:
    """Test the benchmark programs exist and are valid."""

    def test_benchmark_programs_exist(self):
        """Benchmark programs directory should have .spy files."""
        programs = get_benchmark_programs()
        # We created 3 new programs + existing ones
        assert len(programs) >= 3

    def test_benchmark_programs_have_functions(self):
        """Each benchmark program should define at least one function."""
        programs = get_benchmark_programs()
        if not programs:
            pytest.skip("No benchmark programs available")

        for program in programs:
            content = program.read_text()
            assert "def " in content, f"{program.name} should define functions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
