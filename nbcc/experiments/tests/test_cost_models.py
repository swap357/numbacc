"""Unit tests for cost models."""

import pytest

from nbcc.experiments.cost_models import (
    BaselineCostModel,
    CostModelStats,
    HybridCostModel,
    MLCostModel,
    PerfCostModel,
    get_cost_model,
)
from nbcc.experiments.features import (
    NodeFeatures,
    OpCategory,
    extract_features,
)


class TestBaselineCostModel:
    """Tests for BaselineCostModel."""

    def test_python_ops_expensive(self):
        """Python generic ops should have high cost."""
        model = BaselineCostModel()

        cost_func = model.get_cost_function(
            "node1", "Py_Call", "Unit", 1.0, []
        )
        # Cost should be very high
        cost = cost_func.compute()
        assert cost >= 10000

    def test_lowered_ops_cheap(self):
        """Lowered/specialized ops should have low cost."""
        model = BaselineCostModel()

        cost_func = model.get_cost_function(
            "node1", "Op_i32_add", "i32", 1.0, []
        )
        cost = cost_func.compute()
        assert cost <= 10

    def test_callfqn_cheap(self):
        """CallFQN should be cheap."""
        model = BaselineCostModel()

        cost_func = model.get_cost_function(
            "node1", "CallFQN", "Unit", 1.0, []
        )
        cost = cost_func.compute()
        assert cost == 1

    def test_stats_tracking(self):
        """Cost model should track statistics."""
        model = BaselineCostModel()

        model.get_cost_function("n1", "Op_i32_add", "i32", 1.0, [])
        model.get_cost_function("n2", "Op_i32_add", "i32", 1.0, [])
        model.get_cost_function("n3", "Py_Call", "Unit", 1.0, [])

        assert model.stats.total_calls == 3
        assert model.stats.ops_evaluated["Op_i32_add"] == 2
        assert model.stats.ops_evaluated["Py_Call"] == 1

    def test_reset_stats(self):
        """Stats should be resettable."""
        model = BaselineCostModel()

        model.get_cost_function("n1", "Op_i32_add", "i32", 1.0, [])
        model.reset_stats()

        assert model.stats.total_calls == 0
        assert len(model.stats.ops_evaluated) == 0


class TestMLCostModel:
    """Tests for MLCostModel."""

    def test_heuristic_prediction(self):
        """ML model should produce reasonable costs without trained model."""
        model = MLCostModel()

        # Arithmetic ops should be cheap
        cost_arith = model.get_cost_function(
            "n1", "Op_i32_add", "i32", 1.0, []
        )
        arith_cost = cost_arith.compute()

        # Python calls should be expensive
        cost_py = model.get_cost_function(
            "n2", "Py_Call", "Unit", 1.0, []
        )
        py_cost = cost_py.compute()

        assert py_cost > arith_cost

    def test_extract_features(self):
        """Feature extraction should work correctly."""
        model = MLCostModel()

        features = model.extract_features("Op_i32_add", "i32", 2)
        assert len(features) > 0
        assert features[0] == OpCategory.ARITHMETIC  # op_category
        assert features[1] == 2  # num_children


class TestPerfCostModel:
    """Tests for PerfCostModel."""

    def test_initialization_no_calibrate(self):
        """Model should initialize without calibration."""
        model = PerfCostModel(calibrate=False)
        assert len(model.measured) == 0

    def test_fallback_costs(self):
        """Model should provide fallback costs for unmeasured ops."""
        model = PerfCostModel(calibrate=False)

        # Without calibration, should still return valid costs
        cost_func = model.get_cost_function(
            "n1", "Py_Call", "Unit", 1.0, []
        )
        cost = cost_func.compute()
        assert cost > 0


class TestHybridCostModel:
    """Tests for HybridCostModel."""

    def test_weight_validation(self):
        """Weights should sum to 1."""
        # Valid weights
        model = HybridCostModel(perf_weight=0.6, ml_weight=0.4)
        assert model.perf_weight + model.ml_weight == 1.0

        # Invalid weights should raise
        with pytest.raises(AssertionError):
            HybridCostModel(perf_weight=0.5, ml_weight=0.3)

    def test_hybrid_cost(self):
        """Hybrid model should produce valid costs."""
        model = HybridCostModel(
            perf_model=PerfCostModel(calibrate=False),
            ml_model=MLCostModel(),
        )

        cost_func = model.get_cost_function(
            "n1", "Op_i32_add", "i32", 1.0, []
        )
        cost = cost_func.compute()
        assert cost > 0


class TestGetCostModel:
    """Tests for get_cost_model factory."""

    def test_baseline(self):
        """Should return BaselineCostModel."""
        model = get_cost_model("baseline")
        assert isinstance(model, BaselineCostModel)

    def test_ml(self):
        """Should return MLCostModel."""
        model = get_cost_model("ml")
        assert isinstance(model, MLCostModel)

        model2 = get_cost_model("xgboost")
        assert isinstance(model2, MLCostModel)

    def test_unknown(self):
        """Should raise for unknown model."""
        with pytest.raises(ValueError):
            get_cost_model("unknown_model")


class TestFeatures:
    """Tests for feature extraction."""

    def test_extract_features(self):
        """Feature extraction should work correctly."""
        features = extract_features("Op_i32_add", "i32", 2)

        assert features.op_category == OpCategory.ARITHMETIC
        assert features.num_children == 2
        assert features.is_pure is True
        assert features.has_io is False
        assert features.is_lowered is True
        assert features.is_python_generic is False

    def test_python_op_features(self):
        """Python ops should have correct features."""
        features = extract_features("Py_Call", "Unit", 1)

        assert features.op_category == OpCategory.CALL
        assert features.is_python_generic is True
        assert features.has_io is True  # Py_Call may have IO
        assert features.is_pure is False

    def test_io_op_features(self):
        """IO ops should be marked correctly."""
        features = extract_features("Builtin_print_i32", "Unit", 2)

        assert features.op_category == OpCategory.MEMORY_IO
        assert features.has_io is True

    def test_to_vector(self):
        """Features should convert to vector correctly."""
        features = extract_features("Op_i32_add", "i32", 2)
        vector = features.to_vector()

        assert isinstance(vector, list)
        assert len(vector) == len(NodeFeatures.feature_names())
        assert all(isinstance(v, float) for v in vector)
