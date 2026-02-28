"""
Extraction Analysis Module for E-Graph Cost Model Experiments.

Provides instrumentation to see what happens inside extraction:
- Per-eclass alternatives and their costs
- Which alternative was chosen and why
- Agreement/disagreement between cost models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from egglog import EGraph
from sealir.eqsat.rvsdg_extract import (
    Bucket,
    CostModel as _CostModel,
    Extraction,
    egraph_extraction,
)


@dataclass
class EClassAlternatives:
    """
    Represents all alternatives within an equivalence class.

    Attributes:
        eclass: The equivalence class identifier
        alternatives: Mapping from node_name to its cost
        chosen: The node_name that was selected
        chosen_cost: The cost of the chosen node
    """
    eclass: str
    alternatives: dict[str, float]
    chosen: str
    chosen_cost: float

    @property
    def num_alternatives(self) -> int:
        """Number of alternatives in this e-class."""
        return len(self.alternatives)

    @property
    def has_multiple_alternatives(self) -> bool:
        """True if there are multiple choices (where cost model matters)."""
        return len(self.alternatives) > 1

    @property
    def cost_range(self) -> tuple[float, float]:
        """Return (min_cost, max_cost) of alternatives."""
        if not self.alternatives:
            return (float('inf'), float('inf'))
        costs = list(self.alternatives.values())
        return (min(costs), max(costs))

    @property
    def cost_spread(self) -> float:
        """Difference between max and min cost (0 means all equal)."""
        min_c, max_c = self.cost_range
        if min_c == float('inf'):
            return 0.0
        return max_c - min_c


@dataclass
class ExtractionAnalysis:
    """
    Complete analysis of an extraction result.

    Attributes:
        model_name: Name of the cost model used
        total_cost: Total cost of the extracted program
        node_count: Number of nodes in the extracted graph
        eclass_analyses: Per-eclass analysis of alternatives
        stats: Additional statistics from extraction
    """
    model_name: str
    total_cost: float
    node_count: int
    eclass_analyses: list[EClassAlternatives]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def num_eclasses(self) -> int:
        """Total number of e-classes analyzed."""
        return len(self.eclass_analyses)

    @property
    def eclasses_with_choices(self) -> list[EClassAlternatives]:
        """E-classes where there were multiple alternatives to choose from."""
        return [ec for ec in self.eclass_analyses if ec.has_multiple_alternatives]

    @property
    def num_eclasses_with_choices(self) -> int:
        """Number of e-classes where the cost model actually mattered."""
        return len(self.eclasses_with_choices)

    @property
    def total_alternatives(self) -> int:
        """Total number of alternatives across all e-classes."""
        return sum(ec.num_alternatives for ec in self.eclass_analyses)

    @property
    def avg_alternatives_per_eclass(self) -> float:
        """Average number of alternatives per e-class."""
        if not self.eclass_analyses:
            return 0.0
        return self.total_alternatives / len(self.eclass_analyses)

    def get_interesting_eclasses(self, min_spread: float = 0.1) -> list[EClassAlternatives]:
        """
        Get e-classes where alternatives have meaningfully different costs.

        Args:
            min_spread: Minimum cost difference to be considered interesting

        Returns:
            List of EClassAlternatives with cost_spread >= min_spread
        """
        return [
            ec for ec in self.eclass_analyses
            if ec.cost_spread >= min_spread
        ]


@dataclass
class ModelComparison:
    """
    Comparison of extraction results from multiple cost models.

    Attributes:
        program: Name of the program analyzed
        analyses: Dict mapping model_name to ExtractionAnalysis
        disagreements: E-classes where models chose different nodes
    """
    program: str
    analyses: dict[str, ExtractionAnalysis]
    disagreements: list[EClassDisagreement] = field(default_factory=list)

    @property
    def model_names(self) -> list[str]:
        """Names of all models in this comparison."""
        return list(self.analyses.keys())

    @property
    def num_disagreements(self) -> int:
        """Number of e-classes where models disagreed."""
        return len(self.disagreements)

    @property
    def agreement_rate(self) -> float:
        """Fraction of e-classes where all models agreed (0 to 1)."""
        if not self.analyses:
            return 1.0

        # Get e-class count from first analysis
        first = next(iter(self.analyses.values()))
        total_eclasses = first.num_eclasses_with_choices

        if total_eclasses == 0:
            return 1.0

        return 1.0 - (self.num_disagreements / total_eclasses)


@dataclass
class EClassDisagreement:
    """
    Records when different cost models chose different nodes for an e-class.

    Attributes:
        eclass: The equivalence class identifier
        choices: Dict mapping model_name to (chosen_node, cost)
        alternatives: All available alternatives with their costs per model
    """
    eclass: str
    choices: dict[str, tuple[str, float]]  # model_name -> (chosen_node, cost)
    alternatives: dict[str, dict[str, float]]  # model_name -> {node -> cost}

    @property
    def chosen_nodes(self) -> set[str]:
        """Set of all nodes chosen by at least one model."""
        return {choice[0] for choice in self.choices.values()}

    @property
    def num_different_choices(self) -> int:
        """Number of distinct nodes chosen across models."""
        return len(self.chosen_nodes)


def analyze_extraction(
    egraph: EGraph,
    cost_model: _CostModel,
    model_name: str = "unknown",
) -> ExtractionAnalysis:
    """
    Perform extraction and return detailed analysis.

    Args:
        egraph: The e-graph to extract from
        cost_model: The cost model to use
        model_name: Name to identify this model in results

    Returns:
        ExtractionAnalysis with per-eclass details
    """
    stats: dict[str, Any] = {}

    # Run extraction
    extraction = egraph_extraction(egraph, cost_model=cost_model, stats=stats)
    extraction.compute()
    result = extraction.extract_common_root()

    # Analyze selections
    eclass_analyses = []

    if extraction._selections is not None:
        for eclass, bucket in extraction._selections.items():
            # Get all alternatives from this bucket
            alternatives = get_bucket_alternatives(bucket)

            if alternatives:
                best_name, best_cost = bucket.best()
                eclass_analyses.append(EClassAlternatives(
                    eclass=eclass,
                    alternatives=alternatives,
                    chosen=best_name,
                    chosen_cost=best_cost,
                ))

    return ExtractionAnalysis(
        model_name=model_name,
        total_cost=result.cost,
        node_count=result.graph.number_of_nodes(),
        eclass_analyses=eclass_analyses,
        stats=stats,
    )


def get_bucket_alternatives(bucket: Bucket) -> dict[str, float]:
    """
    Extract all alternatives from a Bucket.

    Args:
        bucket: The bucket containing node choices

    Returns:
        Dict mapping node_name to cost
    """
    # Use the Bucket's public method
    return bucket.all_alternatives()


def compare_models(
    egraph: EGraph,
    models: dict[str, _CostModel],
    program_name: str = "unknown",
) -> ModelComparison:
    """
    Compare how different cost models extract from the same e-graph.

    Args:
        egraph: The e-graph to extract from
        models: Dict mapping model_name to CostModel instance
        program_name: Name of the program being analyzed

    Returns:
        ModelComparison with analysis per model and disagreements
    """
    analyses: dict[str, ExtractionAnalysis] = {}

    for name, model in models.items():
        analyses[name] = analyze_extraction(egraph, model, model_name=name)

    # Find disagreements
    disagreements = find_disagreements(analyses)

    return ModelComparison(
        program=program_name,
        analyses=analyses,
        disagreements=disagreements,
    )


def find_disagreements(
    analyses: dict[str, ExtractionAnalysis],
) -> list[EClassDisagreement]:
    """
    Find e-classes where cost models chose different nodes.

    Args:
        analyses: Dict mapping model_name to ExtractionAnalysis

    Returns:
        List of EClassDisagreement for disagreeing e-classes
    """
    if not analyses:
        return []

    disagreements = []

    # Build per-eclass data structure
    eclass_data: dict[str, dict[str, EClassAlternatives]] = {}

    for model_name, analysis in analyses.items():
        for eclass_analysis in analysis.eclass_analyses:
            if eclass_analysis.eclass not in eclass_data:
                eclass_data[eclass_analysis.eclass] = {}
            eclass_data[eclass_analysis.eclass][model_name] = eclass_analysis

    # Find disagreements
    for eclass, per_model in eclass_data.items():
        if len(per_model) < 2:
            continue

        # Get chosen nodes per model
        chosen_nodes = {
            model: analysis.chosen
            for model, analysis in per_model.items()
        }

        # Check if there's disagreement
        unique_choices = set(chosen_nodes.values())
        if len(unique_choices) > 1:
            choices = {
                model: (analysis.chosen, analysis.chosen_cost)
                for model, analysis in per_model.items()
            }
            alternatives = {
                model: analysis.alternatives
                for model, analysis in per_model.items()
            }

            disagreements.append(EClassDisagreement(
                eclass=eclass,
                choices=choices,
                alternatives=alternatives,
            ))

    return disagreements


def print_extraction_report(analysis: ExtractionAnalysis) -> None:
    """
    Print a detailed report of extraction analysis.

    Args:
        analysis: The ExtractionAnalysis to report
    """
    print(f"\n{'='*60}")
    print(f"Extraction Analysis: {analysis.model_name}")
    print(f"{'='*60}")

    print(f"\nSummary:")
    print(f"  Total cost: {analysis.total_cost:.2f}")
    print(f"  Node count: {analysis.node_count}")
    print(f"  E-classes: {analysis.num_eclasses}")
    print(f"  E-classes with choices: {analysis.num_eclasses_with_choices}")
    print(f"  Total alternatives: {analysis.total_alternatives}")
    print(f"  Avg alternatives/eclass: {analysis.avg_alternatives_per_eclass:.2f}")

    # Show interesting e-classes
    interesting = analysis.get_interesting_eclasses(min_spread=0.5)
    if interesting:
        print(f"\nInteresting E-Classes (cost spread >= 0.5):")
        for ec in interesting[:10]:  # Limit to first 10
            print(f"\n  {ec.eclass}:")
            print(f"    Alternatives: {ec.num_alternatives}")
            print(f"    Cost range: {ec.cost_range[0]:.2f} - {ec.cost_range[1]:.2f}")
            print(f"    Chosen: {ec.chosen} (cost={ec.chosen_cost:.2f})")

            # Show top alternatives
            sorted_alts = sorted(ec.alternatives.items(), key=lambda x: x[1])
            print(f"    Top alternatives:")
            for name, cost in sorted_alts[:5]:
                marker = " *" if name == ec.chosen else ""
                print(f"      {name}: {cost:.2f}{marker}")

    if analysis.stats:
        print(f"\nExtraction stats:")
        for key, value in analysis.stats.items():
            print(f"  {key}: {value}")


def print_comparison_report(comparison: ModelComparison) -> None:
    """
    Print a comparison report across multiple cost models.

    Args:
        comparison: The ModelComparison to report
    """
    print(f"\n{'='*60}")
    print(f"Cost Model Comparison: {comparison.program}")
    print(f"{'='*60}")

    print(f"\nModel Summary:")
    print(f"{'Model':<20} {'Total Cost':>12} {'Nodes':>8} {'E-Classes w/Choice':>20}")
    print(f"{'-'*60}")

    for name, analysis in comparison.analyses.items():
        print(f"{name:<20} {analysis.total_cost:>12.2f} {analysis.node_count:>8} "
              f"{analysis.num_eclasses_with_choices:>20}")

    print(f"\nAgreement rate: {comparison.agreement_rate:.1%}")
    print(f"Disagreements: {comparison.num_disagreements}")

    if comparison.disagreements:
        print(f"\nDisagreements (showing first 5):")
        for dis in comparison.disagreements[:5]:
            print(f"\n  E-class: {dis.eclass}")
            print(f"  Choices:")
            for model, (node, cost) in dis.choices.items():
                print(f"    {model}: {node} (cost={cost:.2f})")


def get_extraction_summary(
    egraph: EGraph,
    models: dict[str, _CostModel],
) -> dict[str, Any]:
    """
    Get a quick summary of extraction differences between models.

    Args:
        egraph: The e-graph to extract from
        models: Dict mapping model_name to CostModel instance

    Returns:
        Summary dict with key metrics
    """
    comparison = compare_models(egraph, models)

    summary = {
        "num_models": len(models),
        "model_names": list(models.keys()),
        "agreement_rate": comparison.agreement_rate,
        "num_disagreements": comparison.num_disagreements,
        "costs": {
            name: analysis.total_cost
            for name, analysis in comparison.analyses.items()
        },
        "node_counts": {
            name: analysis.node_count
            for name, analysis in comparison.analyses.items()
        },
        "eclasses_with_choices": {
            name: analysis.num_eclasses_with_choices
            for name, analysis in comparison.analyses.items()
        },
    }

    return summary
