"""
Feature extraction for ML-based cost models.

Extracts features from e-graph nodes for use in cost prediction models.
Features are designed to capture cost-relevant properties of operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class OpCategory(IntEnum):
    """Operation category encoding for ML features."""
    ARITHMETIC = 0
    COMPARISON = 1
    CONTROL = 2
    MEMORY_IO = 3
    CALL = 4
    LITERAL = 5
    UNKNOWN = 6


# Mapping from operation names to categories
OP_TO_CATEGORY: dict[str, OpCategory] = {
    # Arithmetic operations
    "Op_i32_add": OpCategory.ARITHMETIC,
    "Op_i32_sub": OpCategory.ARITHMETIC,
    "Op_i32_mul": OpCategory.ARITHMETIC,
    "Op_i32_div": OpCategory.ARITHMETIC,
    "Op_i32_mod": OpCategory.ARITHMETIC,
    "Op_i32_neg": OpCategory.ARITHMETIC,
    "Op_f64_add": OpCategory.ARITHMETIC,
    "Op_f64_sub": OpCategory.ARITHMETIC,
    "Op_f64_mul": OpCategory.ARITHMETIC,
    "Op_f64_div": OpCategory.ARITHMETIC,
    "Op_f64_neg": OpCategory.ARITHMETIC,
    "Op_bool_and": OpCategory.ARITHMETIC,
    "Op_bool_or": OpCategory.ARITHMETIC,
    "Op_bool_not": OpCategory.ARITHMETIC,

    # Comparison operations
    "Op_i32_lt": OpCategory.COMPARISON,
    "Op_i32_le": OpCategory.COMPARISON,
    "Op_i32_gt": OpCategory.COMPARISON,
    "Op_i32_ge": OpCategory.COMPARISON,
    "Op_i32_eq": OpCategory.COMPARISON,
    "Op_i32_ne": OpCategory.COMPARISON,
    "Op_f64_lt": OpCategory.COMPARISON,
    "Op_f64_le": OpCategory.COMPARISON,
    "Op_f64_gt": OpCategory.COMPARISON,
    "Op_f64_ge": OpCategory.COMPARISON,
    "Op_f64_eq": OpCategory.COMPARISON,
    "Op_f64_ne": OpCategory.COMPARISON,

    # Control flow operations
    "Gamma": OpCategory.CONTROL,
    "Theta": OpCategory.CONTROL,

    # Memory/IO operations
    "Builtin_print_i32": OpCategory.MEMORY_IO,
    "Builtin_print_f64": OpCategory.MEMORY_IO,
    "Builtin_print_str": OpCategory.MEMORY_IO,

    # Call operations
    "Py_Call": OpCategory.CALL,
    "Py_LoadGlobal": OpCategory.CALL,
    "CallFQN": OpCategory.CALL,

    # Literals
    "Literal": OpCategory.LITERAL,
    "Const": OpCategory.LITERAL,
}

# Operations that touch IO state
IO_OPS: frozenset[str] = frozenset({
    "Builtin_print_i32",
    "Builtin_print_f64",
    "Builtin_print_str",
    "Py_Call",  # May have IO side effects
})

# Pure operations (no side effects, deterministic)
PURE_OPS: frozenset[str] = frozenset({
    # Arithmetic
    "Op_i32_add", "Op_i32_sub", "Op_i32_mul", "Op_i32_div", "Op_i32_mod", "Op_i32_neg",
    "Op_f64_add", "Op_f64_sub", "Op_f64_mul", "Op_f64_div", "Op_f64_neg",
    "Op_bool_and", "Op_bool_or", "Op_bool_not",
    # Comparison
    "Op_i32_lt", "Op_i32_le", "Op_i32_gt", "Op_i32_ge", "Op_i32_eq", "Op_i32_ne",
    "Op_f64_lt", "Op_f64_le", "Op_f64_gt", "Op_f64_ge", "Op_f64_eq", "Op_f64_ne",
    # Literals
    "Literal", "Const",
    # Specialized calls (assumed pure)
    "CallFQN",
})

# Type weights for cost estimation
TYPE_WEIGHTS: dict[str, float] = {
    "i32": 1.0,
    "i64": 1.0,
    "f32": 1.2,
    "f64": 1.5,
    "bool": 0.5,
    "str": 2.0,
    "Unit": 0.0,
}


@dataclass
class NodeFeatures:
    """
    Features extracted from an e-graph node for ML prediction.

    Attributes:
        op_category: Category of operation (0-6)
        num_children: Number of child nodes (fan-in)
        has_io: Whether the operation touches IO state
        is_pure: Whether the operation is pure (no side effects)
        type_weight: Weight associated with the result type
        is_python_generic: Whether this is a generic Python operation
        is_lowered: Whether this is a lowered/specialized operation
    """
    op_category: int
    num_children: int
    has_io: bool
    is_pure: bool
    type_weight: float = 1.0
    is_python_generic: bool = False
    is_lowered: bool = False

    def to_vector(self) -> list[float]:
        """Convert features to a numeric vector for ML models."""
        return [
            float(self.op_category),
            float(self.num_children),
            float(self.has_io),
            float(self.is_pure),
            self.type_weight,
            float(self.is_python_generic),
            float(self.is_lowered),
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names for interpretability."""
        return [
            "op_category",
            "num_children",
            "has_io",
            "is_pure",
            "type_weight",
            "is_python_generic",
            "is_lowered",
        ]


def extract_features(
    op: str,
    ty: str,
    num_children: int,
) -> NodeFeatures:
    """
    Extract features from an e-graph node.

    Args:
        op: Operation name (e.g., "Op_i32_add", "Py_Call")
        ty: Type annotation (e.g., "i32", "f64")
        num_children: Number of child nodes

    Returns:
        NodeFeatures dataclass with extracted features
    """
    # Determine operation category
    op_category = OP_TO_CATEGORY.get(op, OpCategory.UNKNOWN)

    # Check IO and purity
    has_io = op in IO_OPS
    is_pure = op in PURE_OPS

    # Get type weight
    type_weight = TYPE_WEIGHTS.get(ty, 1.0)

    # Check if Python generic or lowered
    is_python_generic = op.startswith("Py_")
    is_lowered = op.startswith("Op_") or op.startswith("Builtin_")

    return NodeFeatures(
        op_category=int(op_category),
        num_children=num_children,
        has_io=has_io,
        is_pure=is_pure,
        type_weight=type_weight,
        is_python_generic=is_python_generic,
        is_lowered=is_lowered,
    )


@dataclass
class ExtendedNodeFeatures(NodeFeatures):
    """
    Extended features including subgraph properties.

    Use when more context is available (e.g., during extraction).
    """
    subtree_depth: int = 0
    subtree_size: int = 0
    num_io_descendants: int = 0
    num_control_descendants: int = 0

    def to_vector(self) -> list[float]:
        """Convert features to a numeric vector."""
        base = super().to_vector()
        return base + [
            float(self.subtree_depth),
            float(self.subtree_size),
            float(self.num_io_descendants),
            float(self.num_control_descendants),
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names for interpretability."""
        base = super().feature_names()
        return base + [
            "subtree_depth",
            "subtree_size",
            "num_io_descendants",
            "num_control_descendants",
        ]


def extract_extended_features(
    op: str,
    ty: str,
    num_children: int,
    subtree_depth: int = 0,
    subtree_size: int = 0,
    num_io_descendants: int = 0,
    num_control_descendants: int = 0,
) -> ExtendedNodeFeatures:
    """
    Extract extended features including subgraph properties.

    Args:
        op: Operation name
        ty: Type annotation
        num_children: Number of child nodes
        subtree_depth: Maximum depth of subtree rooted at this node
        subtree_size: Number of nodes in subtree
        num_io_descendants: Number of IO operations in subtree
        num_control_descendants: Number of control flow nodes in subtree

    Returns:
        ExtendedNodeFeatures dataclass
    """
    base = extract_features(op, ty, num_children)

    return ExtendedNodeFeatures(
        op_category=base.op_category,
        num_children=base.num_children,
        has_io=base.has_io,
        is_pure=base.is_pure,
        type_weight=base.type_weight,
        is_python_generic=base.is_python_generic,
        is_lowered=base.is_lowered,
        subtree_depth=subtree_depth,
        subtree_size=subtree_size,
        num_io_descendants=num_io_descendants,
        num_control_descendants=num_control_descendants,
    )


class FeatureCollector:
    """
    Collects features and costs from compilation for training data.

    Usage:
        collector = FeatureCollector()

        # During compilation
        for node in egraph_nodes:
            features = extract_features(node.op, node.ty, len(node.children))
            collector.add(features, measured_cost)

        # After collecting data
        X, y = collector.to_training_data()
    """

    def __init__(self) -> None:
        self._features: list[list[float]] = []
        self._costs: list[float] = []
        self._metadata: list[dict[str, Any]] = []

    def add(
        self,
        features: NodeFeatures,
        cost: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a training sample."""
        self._features.append(features.to_vector())
        self._costs.append(cost)
        self._metadata.append(metadata or {})

    def to_training_data(self) -> tuple[list[list[float]], list[float]]:
        """Return features and costs as training data."""
        return self._features, self._costs

    def __len__(self) -> int:
        return len(self._features)

    def save(self, path: str) -> None:
        """Save collected data to a JSON file."""
        import json
        with open(path, "w") as f:
            json.dump({
                "features": self._features,
                "costs": self._costs,
                "metadata": self._metadata,
                "feature_names": NodeFeatures.feature_names(),
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureCollector":
        """Load collected data from a JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)

        collector = cls()
        collector._features = data["features"]
        collector._costs = data["costs"]
        collector._metadata = data.get("metadata", [{} for _ in data["features"]])
        return collector
