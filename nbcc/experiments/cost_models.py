"""
Cost model implementations for e-graph extraction experiments.

This module provides multiple cost model strategies to compare their effectiveness
at guiding e-graph extraction toward faster generated code.

Cost Models:
- BaselineCostModel: Static costs per operation type (current approach)
- PerfCostModel: Hardware counter-based measurements
- MLCostModel: XGBoost-based predictions from node features
- HybridCostModel: Weighted combination of Perf and ML models
"""

from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from sealir.eqsat.rvsdg_extract import CostFunc, CostModel as _CostModel

if TYPE_CHECKING:
    from .features import NodeFeatures


@dataclass
class CostModelStats:
    """Statistics collected during cost model evaluation."""
    total_calls: int = 0
    cache_hits: int = 0
    ops_evaluated: dict[str, int] = field(default_factory=dict)


class BaselineCostModel(_CostModel):
    """
    Current static cost model - the baseline to beat.

    Assigns fixed costs based on operation type:
    - Python generic operations (Py_Call, Py_LoadGlobal): 10000 (very expensive)
    - Specialized operations (CallFQN): 1 (cheap)
    - Arithmetic operations (Op_i32_add, etc.): 1
    - IO operations (Builtin_print_i32): 10

    This guides extraction toward specialized lowered operations over
    generic Python operations.
    """

    COSTS: dict[str, float] = {
        # Python generic operations - very expensive (should be avoided)
        "Py_Call": 10000,
        "Py_LoadGlobal": 10000,

        # Specialized operations - cheap (preferred)
        "CallFQN": 1,

        # i32 arithmetic operations
        "Op_i32_add": 1,
        "Op_i32_sub": 1,
        "Op_i32_mul": 1,
        "Op_i32_div": 1,
        "Op_i32_mod": 1,
        "Op_i32_neg": 1,

        # i32 comparison operations
        "Op_i32_lt": 1,
        "Op_i32_le": 1,
        "Op_i32_gt": 1,
        "Op_i32_ge": 1,
        "Op_i32_eq": 1,
        "Op_i32_ne": 1,

        # f64 arithmetic operations
        "Op_f64_add": 1,
        "Op_f64_sub": 1,
        "Op_f64_mul": 1,
        "Op_f64_div": 1,
        "Op_f64_neg": 1,

        # f64 comparison operations
        "Op_f64_lt": 1,
        "Op_f64_le": 1,
        "Op_f64_gt": 1,
        "Op_f64_ge": 1,
        "Op_f64_eq": 1,
        "Op_f64_ne": 1,

        # Boolean operations
        "Op_bool_and": 1,
        "Op_bool_or": 1,
        "Op_bool_not": 1,

        # IO operations
        "Builtin_print_i32": 10,
        "Builtin_print_f64": 10,
        "Builtin_print_str": 10,

        # Control flow (base costs, may be scaled by children)
        "Gamma": 2,  # Conditional
        "Theta": 5,  # Loop

        # Literals and constants
        "Literal": 0,
        "Const": 0,
    }

    def __init__(self) -> None:
        self.stats = CostModelStats()

    def get_cost_function(
        self,
        nodename: str,
        op: str,
        ty: str,
        cost: float,
        children: Sequence[str],
    ) -> CostFunc:
        self.stats.total_calls += 1
        self.stats.ops_evaluated[op] = self.stats.ops_evaluated.get(op, 0) + 1

        if op in self.COSTS:
            return self.get_simple(self.COSTS[op])
        elif op in ["Py_Call", "Py_LoadGlobal"]:
            # Catch-all for Python operations
            return self.get_simple(10000)
        elif op in ["CallFQN"]:
            return self.get_simple(1)
        else:
            # Fallback to parent implementation
            return super().get_cost_function(nodename, op, ty, cost, children)

    def reset_stats(self) -> None:
        """Reset collected statistics."""
        self.stats = CostModelStats()


class PerfCostModel(_CostModel):
    """
    Hardware counter-based cost model using perf measurements.

    Measures actual CPU cycles for primitive operations using `perf stat`.
    Falls back to time.perf_counter() if perf is unavailable.

    Key insight: Perf gives ground truth for primitive ops but cannot
    predict composite operation costs.
    """

    # Microbenchmark code templates for each operation type
    MICROBENCHMARKS: dict[str, str] = {
        "Op_i32_add": "x = 1; y = 2\nfor _ in range({n}): z = x + y",
        "Op_i32_sub": "x = 1; y = 2\nfor _ in range({n}): z = x - y",
        "Op_i32_mul": "x = 3; y = 4\nfor _ in range({n}): z = x * y",
        "Op_i32_div": "x = 10; y = 3\nfor _ in range({n}): z = x // y",
        "Op_i32_lt": "x = 1; y = 2\nfor _ in range({n}): z = x < y",
        "Op_i32_gt": "x = 2; y = 1\nfor _ in range({n}): z = x > y",
        "Op_f64_add": "x = 1.0; y = 2.0\nfor _ in range({n}): z = x + y",
        "Op_f64_sub": "x = 1.0; y = 2.0\nfor _ in range({n}): z = x - y",
        "Op_f64_mul": "x = 3.0; y = 4.0\nfor _ in range({n}): z = x * y",
        "Op_f64_div": "x = 10.0; y = 3.0\nfor _ in range({n}): z = x / y",
        "Py_Call": "def f(): pass\nfor _ in range({n}): f()",
        "Py_LoadGlobal": "g = 42\nfor _ in range({n}): x = g",
    }

    def __init__(self, calibrate: bool = True, iterations: int = 100000) -> None:
        self.measured: dict[str, float] = {}
        self.iterations = iterations
        self.stats = CostModelStats()
        self._perf_available = self._check_perf_available()

        if calibrate:
            self._calibrate()

    def _check_perf_available(self) -> bool:
        """Check if perf is available on this system."""
        try:
            result = subprocess.run(
                ["perf", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _measure_with_perf(self, code: str) -> float | None:
        """Measure cycles using perf stat."""
        if not self._perf_available:
            return None

        try:
            result = subprocess.run(
                ["perf", "stat", "-e", "cycles", "-x", ",",
                 "python", "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Parse CSV output: cycles,,cycles,...
            for line in result.stderr.split("\n"):
                if "cycles" in line:
                    parts = line.split(",")
                    if parts[0].isdigit():
                        return float(parts[0])
        except (subprocess.TimeoutExpired, ValueError):
            pass
        return None

    def _measure_with_time(self, code: str) -> float:
        """Fallback: measure with time.perf_counter()."""
        # Wrap in timing code
        timing_code = f"""
import time
start = time.perf_counter_ns()
{code}
end = time.perf_counter_ns()
print(end - start)
"""
        try:
            result = subprocess.run(
                ["python", "-c", timing_code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError):
            return float("inf")

    def _measure_op(self, op: str) -> float:
        """Measure the cost of an operation type."""
        if op not in self.MICROBENCHMARKS:
            return 1.0  # Default cost for unknown ops

        code = self.MICROBENCHMARKS[op].format(n=self.iterations)

        # Try perf first, fall back to time
        cycles = self._measure_with_perf(code)
        if cycles is not None:
            return cycles / self.iterations

        # Fallback: use wall-clock time (in nanoseconds)
        ns = self._measure_with_time(code)
        return ns / self.iterations

    def _calibrate(self) -> None:
        """Run microbenchmarks to measure actual operation costs."""
        for op in self.MICROBENCHMARKS:
            self.measured[op] = self._measure_op(op)

        # Normalize: scale so that Op_i32_add = 1.0
        baseline = self.measured.get("Op_i32_add", 1.0)
        if baseline > 0:
            for op in self.measured:
                self.measured[op] /= baseline

    def get_cost_function(
        self,
        nodename: str,
        op: str,
        ty: str,
        cost: float,
        children: Sequence[str],
    ) -> CostFunc:
        self.stats.total_calls += 1
        self.stats.ops_evaluated[op] = self.stats.ops_evaluated.get(op, 0) + 1

        if op in self.measured:
            return self.get_simple(self.measured[op])
        elif op in ["Py_Call", "Py_LoadGlobal"]:
            # Use measured Python call overhead if available
            py_cost = self.measured.get("Py_Call", 10000)
            return self.get_simple(py_cost)
        elif op in ["CallFQN"]:
            return self.get_simple(1)
        else:
            return super().get_cost_function(nodename, op, ty, cost, children)

    def reset_stats(self) -> None:
        """Reset collected statistics."""
        self.stats = CostModelStats()


class MLCostModel(_CostModel):
    """
    Machine learning-based cost model using XGBoost.

    Predicts operation costs from node features:
    - Operation category (arithmetic, control, memory, call, literal)
    - Number of children (fan-in)
    - Whether the node touches IO state
    - Whether the node is pure (no side effects)

    Training data can come from:
    - Synthetic programs with measured runtimes
    - Online collection from real compilations
    """

    # Operation category encoding
    OP_CATEGORIES: dict[str, int] = {
        # Arithmetic = 0
        "Op_i32_add": 0, "Op_i32_sub": 0, "Op_i32_mul": 0, "Op_i32_div": 0,
        "Op_i32_mod": 0, "Op_i32_neg": 0,
        "Op_f64_add": 0, "Op_f64_sub": 0, "Op_f64_mul": 0, "Op_f64_div": 0,
        "Op_f64_neg": 0,
        "Op_bool_and": 0, "Op_bool_or": 0, "Op_bool_not": 0,
        # Comparison = 1
        "Op_i32_lt": 1, "Op_i32_le": 1, "Op_i32_gt": 1, "Op_i32_ge": 1,
        "Op_i32_eq": 1, "Op_i32_ne": 1,
        "Op_f64_lt": 1, "Op_f64_le": 1, "Op_f64_gt": 1, "Op_f64_ge": 1,
        "Op_f64_eq": 1, "Op_f64_ne": 1,
        # Control = 2
        "Gamma": 2, "Theta": 2,
        # Memory/IO = 3
        "Builtin_print_i32": 3, "Builtin_print_f64": 3, "Builtin_print_str": 3,
        # Call = 4
        "Py_Call": 4, "CallFQN": 4, "Py_LoadGlobal": 4,
        # Literal = 5
        "Literal": 5, "Const": 5,
    }

    # Operations with IO side effects
    IO_OPS: set[str] = {
        "Builtin_print_i32", "Builtin_print_f64", "Builtin_print_str",
        "Py_Call",  # May have IO
    }

    # Pure operations (no side effects)
    PURE_OPS: set[str] = {
        "Op_i32_add", "Op_i32_sub", "Op_i32_mul", "Op_i32_div", "Op_i32_mod",
        "Op_f64_add", "Op_f64_sub", "Op_f64_mul", "Op_f64_div",
        "Op_i32_lt", "Op_i32_gt", "Op_i32_le", "Op_i32_ge", "Op_i32_eq", "Op_i32_ne",
        "Op_f64_lt", "Op_f64_gt", "Op_f64_le", "Op_f64_ge", "Op_f64_eq", "Op_f64_ne",
        "Op_bool_and", "Op_bool_or", "Op_bool_not",
        "Literal", "Const", "CallFQN",
    }

    def __init__(
        self,
        model_path: str | Path | None = None,
        predictor: str = "xgboost",
    ) -> None:
        self.predictor_type = predictor
        self.model: Any = None
        self.stats = CostModelStats()
        self._feature_extractor: Callable[[str, str, int], list[float]] | None = None

        if model_path is not None:
            self.load_model(model_path)
        else:
            # Initialize with default weights (heuristic-based)
            self._init_default_weights()

    def _init_default_weights(self) -> None:
        """Initialize with heuristic-based weights when no trained model exists."""
        # Coefficients for: [category, num_children, has_io, is_pure]
        self._default_weights = {
            "intercept": 1.0,
            "category": [1.0, 1.0, 3.0, 5.0, 100.0, 0.0],  # Per-category base cost
            "num_children": 0.5,  # Per-child cost
            "has_io": 10.0,  # IO penalty
            "is_pure": -0.5,  # Pure bonus
        }

    def extract_features(
        self,
        op: str,
        ty: str,
        num_children: int,
    ) -> list[float]:
        """Extract features for ML prediction."""
        from .features import NodeFeatures, extract_features

        features = extract_features(op, ty, num_children)
        return features.to_vector()

    def predict(self, features: list[float]) -> float:
        """Predict cost from features."""
        if self.model is not None:
            # Use trained model
            import numpy as np
            X = np.array([features])
            return float(self.model.predict(X)[0])
        else:
            # Use heuristic weights
            return self._predict_heuristic(features)

    def _predict_heuristic(self, features: list[float]) -> float:
        """Fallback prediction using heuristic weights."""
        # features: [op_category, num_children, has_io, is_pure]
        if len(features) < 4:
            return 1.0

        w = self._default_weights
        category = int(features[0])
        num_children = features[1]
        has_io = features[2]
        is_pure = features[3]

        cost = w["intercept"]
        if 0 <= category < len(w["category"]):
            cost += w["category"][category]
        cost += w["num_children"] * num_children
        cost += w["has_io"] * has_io
        cost += w["is_pure"] * is_pure

        return max(0.1, cost)  # Ensure positive cost

    def get_cost_function(
        self,
        nodename: str,
        op: str,
        ty: str,
        cost: float,
        children: Sequence[str],
    ) -> CostFunc:
        self.stats.total_calls += 1
        self.stats.ops_evaluated[op] = self.stats.ops_evaluated.get(op, 0) + 1

        features = self.extract_features(op, ty, len(children))
        predicted_cost = self.predict(features)

        return self.get_simple(predicted_cost)

    def load_model(self, path: str | Path) -> None:
        """Load a trained XGBoost model."""
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor()
            self.model.load_model(str(path))
        except ImportError:
            raise ImportError(
                "XGBoost is required for MLCostModel. "
                "Install with: pip install xgboost"
            )

    def save_model(self, path: str | Path) -> None:
        """Save the trained model."""
        if self.model is not None:
            self.model.save_model(str(path))

    def train(
        self,
        features: list[list[float]],
        costs: list[float],
        **kwargs: Any,
    ) -> None:
        """Train the XGBoost model on provided data."""
        try:
            import numpy as np
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost and numpy are required for training. "
                "Install with: pip install xgboost numpy"
            )

        X = np.array(features)
        y = np.array(costs)

        # Default XGBoost parameters for small datasets
        default_params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
        }
        default_params.update(kwargs)

        self.model = xgb.XGBRegressor(**default_params)
        self.model.fit(X, y)

    def reset_stats(self) -> None:
        """Reset collected statistics."""
        self.stats = CostModelStats()


class HybridCostModel(_CostModel):
    """
    Hybrid cost model combining Perf measurements with ML predictions.

    Strategy:
    - For operations with perf measurements: weighted average of perf and ML
    - For operations without measurements: pure ML prediction

    Default weights: 0.7 * perf + 0.3 * ML (when perf is available)
    """

    def __init__(
        self,
        perf_weight: float = 0.7,
        ml_weight: float = 0.3,
        perf_model: PerfCostModel | None = None,
        ml_model: MLCostModel | None = None,
    ) -> None:
        assert abs(perf_weight + ml_weight - 1.0) < 0.001, "Weights must sum to 1"

        self.perf_weight = perf_weight
        self.ml_weight = ml_weight
        self.perf_model = perf_model or PerfCostModel(calibrate=True)
        self.ml_model = ml_model or MLCostModel()
        self.stats = CostModelStats()

    def get_cost_function(
        self,
        nodename: str,
        op: str,
        ty: str,
        cost: float,
        children: Sequence[str],
    ) -> CostFunc:
        self.stats.total_calls += 1
        self.stats.ops_evaluated[op] = self.stats.ops_evaluated.get(op, 0) + 1

        # Get ML prediction
        features = self.ml_model.extract_features(op, ty, len(children))
        ml_cost = self.ml_model.predict(features)

        # Check if we have perf measurements for this op
        if op in self.perf_model.measured:
            perf_cost = self.perf_model.measured[op]
            combined = self.perf_weight * perf_cost + self.ml_weight * ml_cost
            return self.get_simple(combined)
        else:
            # No perf data, use pure ML
            return self.get_simple(ml_cost)

    def reset_stats(self) -> None:
        """Reset collected statistics."""
        self.stats = CostModelStats()
        self.perf_model.reset_stats()
        self.ml_model.reset_stats()


def get_cost_model(name: str, **kwargs: Any) -> _CostModel:
    """
    Factory function to get a cost model by name.

    Args:
        name: One of "baseline", "perf", "xgboost", "hybrid"
        **kwargs: Additional arguments passed to the cost model constructor

    Returns:
        A CostModel instance
    """
    models = {
        "baseline": BaselineCostModel,
        "perf": PerfCostModel,
        "xgboost": MLCostModel,
        "ml": MLCostModel,
        "hybrid": HybridCostModel,
    }

    if name not in models:
        raise ValueError(
            f"Unknown cost model: {name}. "
            f"Available: {list(models.keys())}"
        )

    return models[name](**kwargs)
