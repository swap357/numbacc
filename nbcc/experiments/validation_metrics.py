"""
Validation Metrics for Cost Model Evaluation.

This module provides metrics to validate how well cost models predict
actual runtime performance. The key insight is that we care about
RELATIVE ordering, not absolute values.

Key Metrics:
- Spearman Correlation: Rank correlation between predicted and actual
- Accuracy: Percentage of correct pairwise orderings
- Mean Absolute Error (MAE): Average error in cost predictions

Interpretation:
- Spearman > 0.7: Good - model captures relative performance well
- Spearman 0.3-0.7: Moderate - some predictive power
- Spearman < 0.3: Poor - model is unreliable
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ValidationResult:
    """
    Validation results for a cost model.

    Attributes:
        model_name: Name of the cost model
        spearman_correlation: Rank correlation (-1 to 1)
        kendall_tau: Kendall's tau rank correlation (-1 to 1)
        accuracy: Fraction of correct pairwise orderings (0 to 1)
        mae: Mean Absolute Error (normalized)
        rmse: Root Mean Squared Error (normalized)
        num_samples: Number of samples used
    """
    model_name: str
    spearman_correlation: float
    kendall_tau: float = 0.0
    accuracy: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    num_samples: int = 0

    @property
    def quality_rating(self) -> str:
        """Return a qualitative rating based on Spearman correlation."""
        if self.spearman_correlation > 0.8:
            return "Excellent"
        elif self.spearman_correlation > 0.6:
            return "Good"
        elif self.spearman_correlation > 0.4:
            return "Moderate"
        elif self.spearman_correlation > 0.2:
            return "Weak"
        else:
            return "Poor"


@dataclass
class PairwiseComparison:
    """
    Result of comparing two programs.

    Used to track which orderings the cost model gets right/wrong.
    """
    program_a: str
    program_b: str
    predicted_a_faster: bool  # True if cost_a < cost_b
    actual_a_faster: bool     # True if runtime_a < runtime_b
    is_correct: bool          # True if prediction matches reality
    cost_a: float
    cost_b: float
    runtime_a: float
    runtime_b: float


@dataclass
class DetailedValidation:
    """
    Detailed validation with per-comparison breakdowns.

    Attributes:
        result: Summary validation result
        comparisons: Individual pairwise comparisons
        correct_comparisons: Comparisons where model was correct
        incorrect_comparisons: Comparisons where model was wrong
    """
    result: ValidationResult
    comparisons: list[PairwiseComparison] = field(default_factory=list)

    @property
    def correct_comparisons(self) -> list[PairwiseComparison]:
        return [c for c in self.comparisons if c.is_correct]

    @property
    def incorrect_comparisons(self) -> list[PairwiseComparison]:
        return [c for c in self.comparisons if not c.is_correct]

    @property
    def tie_comparisons(self) -> list[PairwiseComparison]:
        """Comparisons where costs were equal (neither a nor b predicted faster)."""
        return [c for c in self.comparisons
                if c.cost_a == c.cost_b]


def compute_spearman(
    predicted: Sequence[float],
    actual: Sequence[float],
) -> float:
    """
    Compute Spearman rank correlation coefficient.

    Args:
        predicted: Predicted costs from cost model
        actual: Actual runtimes

    Returns:
        Spearman correlation coefficient (-1 to 1)
        1.0 = perfect correlation (cost model is perfect)
        0.0 = no correlation (cost model is random)
        -1.0 = inverse correlation (cost model is backwards)
    """
    if len(predicted) != len(actual) or len(predicted) < 2:
        return 0.0

    # Try scipy first
    try:
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(predicted, actual)
        if math.isnan(correlation):
            return 0.0
        return float(correlation)
    except ImportError:
        pass

    # Fallback: manual implementation
    n = len(predicted)

    def rank(values: Sequence[float]) -> list[float]:
        """Compute ranks with average rank for ties."""
        sorted_indices = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n

        i = 0
        while i < n:
            j = i
            # Find all ties
            while j < n and values[sorted_indices[j]] == values[sorted_indices[i]]:
                j += 1
            # Assign average rank to all ties
            avg_rank = (i + j - 1) / 2.0
            for k in range(i, j):
                ranks[sorted_indices[k]] = avg_rank
            i = j

        return ranks

    rank_predicted = rank(list(predicted))
    rank_actual = rank(list(actual))

    # Spearman correlation formula
    d_squared = sum((rank_predicted[i] - rank_actual[i]) ** 2 for i in range(n))

    # Handle case where all values are the same
    if all(p == predicted[0] for p in predicted) or all(a == actual[0] for a in actual):
        return 0.0

    return 1 - (6 * d_squared) / (n * (n * n - 1))


def compute_kendall_tau(
    predicted: Sequence[float],
    actual: Sequence[float],
) -> float:
    """
    Compute Kendall's tau rank correlation coefficient.

    More robust to outliers than Spearman.

    Args:
        predicted: Predicted costs from cost model
        actual: Actual runtimes

    Returns:
        Kendall's tau (-1 to 1)
    """
    if len(predicted) != len(actual) or len(predicted) < 2:
        return 0.0

    try:
        from scipy.stats import kendalltau
        correlation, p_value = kendalltau(predicted, actual)
        if math.isnan(correlation):
            return 0.0
        return float(correlation)
    except ImportError:
        pass

    # Fallback: manual implementation
    n = len(predicted)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = predicted[i] - predicted[j]
            actual_diff = actual[i] - actual[j]

            if pred_diff * actual_diff > 0:
                concordant += 1
            elif pred_diff * actual_diff < 0:
                discordant += 1
            # Ties are not counted

    total = concordant + discordant
    if total == 0:
        return 0.0

    return (concordant - discordant) / total


def compute_accuracy(
    predicted: Sequence[float],
    actual: Sequence[float],
    programs: Sequence[str] | None = None,
) -> tuple[float, list[PairwiseComparison]]:
    """
    Compute pairwise ordering accuracy.

    For each pair of programs, checks if the cost model correctly
    predicts which one is faster.

    Args:
        predicted: Predicted costs
        actual: Actual runtimes
        programs: Optional program names for reporting

    Returns:
        Tuple of (accuracy, list of PairwiseComparison)
    """
    if len(predicted) != len(actual) or len(predicted) < 2:
        return 0.0, []

    n = len(predicted)
    if programs is None:
        programs = [f"prog_{i}" for i in range(n)]

    comparisons: list[PairwiseComparison] = []
    correct = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Skip if actual times are equal (can't be wrong)
            if actual[i] == actual[j]:
                continue

            predicted_i_faster = predicted[i] < predicted[j]
            actual_i_faster = actual[i] < actual[j]
            is_correct = predicted_i_faster == actual_i_faster

            comparisons.append(PairwiseComparison(
                program_a=programs[i],
                program_b=programs[j],
                predicted_a_faster=predicted_i_faster,
                actual_a_faster=actual_i_faster,
                is_correct=is_correct,
                cost_a=predicted[i],
                cost_b=predicted[j],
                runtime_a=actual[i],
                runtime_b=actual[j],
            ))

            if is_correct:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, comparisons


def compute_mae(
    predicted: Sequence[float],
    actual: Sequence[float],
    normalize: bool = True,
) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        predicted: Predicted costs
        actual: Actual runtimes
        normalize: If True, normalize by mean of actual values

    Returns:
        MAE (lower is better)
    """
    if len(predicted) != len(actual) or len(predicted) == 0:
        return float('inf')

    # Scale predicted to same range as actual for fair comparison
    pred_mean = sum(predicted) / len(predicted)
    actual_mean = sum(actual) / len(actual)

    if pred_mean == 0:
        return float('inf')

    scale = actual_mean / pred_mean
    scaled_predicted = [p * scale for p in predicted]

    errors = [abs(scaled_predicted[i] - actual[i]) for i in range(len(actual))]
    mae = sum(errors) / len(errors)

    if normalize and actual_mean > 0:
        mae /= actual_mean

    return mae


def compute_rmse(
    predicted: Sequence[float],
    actual: Sequence[float],
    normalize: bool = True,
) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        predicted: Predicted costs
        actual: Actual runtimes
        normalize: If True, normalize by mean of actual values

    Returns:
        RMSE (lower is better)
    """
    if len(predicted) != len(actual) or len(predicted) == 0:
        return float('inf')

    # Scale predicted to same range as actual
    pred_mean = sum(predicted) / len(predicted)
    actual_mean = sum(actual) / len(actual)

    if pred_mean == 0:
        return float('inf')

    scale = actual_mean / pred_mean
    scaled_predicted = [p * scale for p in predicted]

    squared_errors = [(scaled_predicted[i] - actual[i]) ** 2 for i in range(len(actual))]
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

    if normalize and actual_mean > 0:
        rmse /= actual_mean

    return rmse


def validate_cost_model(
    predictions: Sequence[float],
    actuals: Sequence[float],
    model_name: str = "unknown",
    programs: Sequence[str] | None = None,
) -> ValidationResult:
    """
    Validate a cost model against actual runtimes.

    Args:
        predictions: Predicted costs from the model
        actuals: Actual measured runtimes
        model_name: Name of the cost model
        programs: Optional program names

    Returns:
        ValidationResult with all metrics
    """
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have same length")

    spearman = compute_spearman(predictions, actuals)
    kendall = compute_kendall_tau(predictions, actuals)
    accuracy, _ = compute_accuracy(predictions, actuals, programs)
    mae = compute_mae(predictions, actuals)
    rmse = compute_rmse(predictions, actuals)

    return ValidationResult(
        model_name=model_name,
        spearman_correlation=spearman,
        kendall_tau=kendall,
        accuracy=accuracy,
        mae=mae,
        rmse=rmse,
        num_samples=len(predictions),
    )


def validate_cost_model_detailed(
    predictions: Sequence[float],
    actuals: Sequence[float],
    model_name: str = "unknown",
    programs: Sequence[str] | None = None,
) -> DetailedValidation:
    """
    Validate a cost model with detailed comparison breakdowns.

    Args:
        predictions: Predicted costs from the model
        actuals: Actual measured runtimes
        model_name: Name of the cost model
        programs: Optional program names

    Returns:
        DetailedValidation with metrics and comparison details
    """
    result = validate_cost_model(predictions, actuals, model_name, programs)
    _, comparisons = compute_accuracy(predictions, actuals, programs)

    return DetailedValidation(result=result, comparisons=comparisons)


def print_validation_report(results: dict[str, ValidationResult]) -> None:
    """
    Print a formatted validation report for multiple cost models.

    Args:
        results: Dict mapping model names to ValidationResult
    """
    print("\n" + "=" * 70)
    print("Cost Model Validation Report")
    print("=" * 70)

    print(f"\n{'Model':<15} {'Spearman':>10} {'Accuracy':>10} {'MAE':>10} {'Rating':<12}")
    print("-" * 57)

    # Sort by Spearman correlation (best first)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].spearman_correlation,
        reverse=True
    )

    for model_name, result in sorted_results:
        print(f"{model_name:<15} {result.spearman_correlation:>10.3f} "
              f"{result.accuracy:>9.1%} {result.mae:>10.4f} "
              f"{result.quality_rating:<12}")

    print("-" * 57)

    # Interpretation
    best_name, best_result = sorted_results[0]
    print(f"\nBest model: {best_name}")
    print(f"  Spearman correlation: {best_result.spearman_correlation:.3f}")
    print(f"  Quality rating: {best_result.quality_rating}")

    if best_result.spearman_correlation < 0.5:
        print("\n  Warning: Even the best model has weak correlation.")
        print("  Cost model predictions may not reliably predict runtime.")
    elif best_result.spearman_correlation > 0.7:
        print(f"\n  The {best_name} model shows good predictive power.")
        print("  Extraction choices guided by this model should produce faster code.")


def print_detailed_validation_report(validation: DetailedValidation) -> None:
    """
    Print detailed validation report with comparison breakdowns.

    Args:
        validation: DetailedValidation to report
    """
    result = validation.result

    print(f"\n{'='*60}")
    print(f"Detailed Validation: {result.model_name}")
    print(f"{'='*60}")

    print(f"\nSummary Metrics:")
    print(f"  Spearman correlation: {result.spearman_correlation:.3f}")
    print(f"  Kendall's tau:        {result.kendall_tau:.3f}")
    print(f"  Pairwise accuracy:    {result.accuracy:.1%}")
    print(f"  MAE (normalized):     {result.mae:.4f}")
    print(f"  RMSE (normalized):    {result.rmse:.4f}")
    print(f"  Samples:              {result.num_samples}")
    print(f"  Rating:               {result.quality_rating}")

    # Show incorrect predictions
    incorrect = validation.incorrect_comparisons
    if incorrect:
        print(f"\nIncorrect predictions ({len(incorrect)} of {len(validation.comparisons)}):")
        for comp in incorrect[:10]:  # Show first 10
            pred = f"{comp.program_a} < {comp.program_b}" if comp.predicted_a_faster else f"{comp.program_b} < {comp.program_a}"
            actual = f"{comp.program_a} < {comp.program_b}" if comp.actual_a_faster else f"{comp.program_b} < {comp.program_a}"
            print(f"  Predicted: {pred}")
            print(f"    Actual:  {actual}")
            print(f"    Costs:   {comp.cost_a:.2f} vs {comp.cost_b:.2f}")
            print(f"    Runtime: {comp.runtime_a:.0f} vs {comp.runtime_b:.0f} ns")
    else:
        print("\nAll pairwise predictions were correct!")


def compare_cost_model_quality(
    model_results: dict[str, ValidationResult],
) -> tuple[str, str]:
    """
    Compare cost models and return the best and worst.

    Args:
        model_results: Dict mapping model names to ValidationResult

    Returns:
        Tuple of (best_model_name, worst_model_name)
    """
    sorted_models = sorted(
        model_results.items(),
        key=lambda x: x[1].spearman_correlation,
        reverse=True
    )

    best = sorted_models[0][0]
    worst = sorted_models[-1][0]

    return best, worst


if __name__ == "__main__":
    # Test the validation metrics with synthetic data
    import random

    # Simulate cost model predictions and actual runtimes
    n = 10
    random.seed(42)

    # Good cost model: predicted ≈ actual + noise
    actual = [random.uniform(100, 1000) for _ in range(n)]
    good_predicted = [a * random.uniform(0.8, 1.2) for a in actual]
    bad_predicted = [random.uniform(100, 1000) for _ in range(n)]

    programs = [f"prog_{i}" for i in range(n)]

    print("Testing validation metrics...")

    good_result = validate_cost_model(good_predicted, actual, "good_model", programs)
    bad_result = validate_cost_model(bad_predicted, actual, "random_model", programs)

    results = {
        "good_model": good_result,
        "random_model": bad_result,
    }

    print_validation_report(results)

    # Detailed report for good model
    detailed = validate_cost_model_detailed(good_predicted, actual, "good_model", programs)
    print_detailed_validation_report(detailed)
