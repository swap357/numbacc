"""
Machine learning training pipeline for cost models.

This module provides:
1. Training data collection from e-graph extraction
2. Feature engineering for cost prediction
3. Model training and validation
4. Model persistence and loading
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .features import NodeFeatures, extract_features
from .principled_costs import OPERATION_COSTS, InstructionCost


@dataclass
class TrainingSample:
    """A single training sample with features and target cost."""
    op: str
    ty: str
    num_children: int
    features: list[float]
    predicted_cost: float  # From analytical model
    actual_cost: float | None = None  # From measurement (if available)

    def target(self) -> float:
        """Return the training target (actual if available, else predicted)."""
        return self.actual_cost if self.actual_cost is not None else self.predicted_cost


@dataclass
class TrainingDataset:
    """Collection of training samples."""
    samples: list[TrainingSample] = field(default_factory=list)

    def add(
        self,
        op: str,
        ty: str,
        num_children: int,
        predicted_cost: float,
        actual_cost: float | None = None,
    ) -> None:
        """Add a training sample."""
        features = extract_features(op, ty, num_children)
        self.samples.append(TrainingSample(
            op=op,
            ty=ty,
            num_children=num_children,
            features=features.to_vector(),
            predicted_cost=predicted_cost,
            actual_cost=actual_cost,
        ))

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays for training."""
        X = np.array([s.features for s in self.samples])
        y = np.array([s.target() for s in self.samples])
        return X, y

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON."""
        data = {
            "samples": [
                {
                    "op": s.op,
                    "ty": s.ty,
                    "num_children": s.num_children,
                    "features": s.features,
                    "predicted_cost": s.predicted_cost,
                    "actual_cost": s.actual_cost,
                }
                for s in self.samples
            ],
            "feature_names": NodeFeatures.feature_names(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TrainingDataset":
        """Load dataset from JSON."""
        with open(path) as f:
            data = json.load(f)

        dataset = cls()
        for s in data["samples"]:
            dataset.samples.append(TrainingSample(
                op=s["op"],
                ty=s["ty"],
                num_children=s["num_children"],
                features=s["features"],
                predicted_cost=s["predicted_cost"],
                actual_cost=s.get("actual_cost"),
            ))
        return dataset

    def __len__(self) -> int:
        return len(self.samples)


def generate_training_data_from_operations() -> TrainingDataset:
    """
    Generate training data from known operation costs.

    This creates synthetic training data based on the analytical
    cost model, which can bootstrap ML training.
    """
    dataset = TrainingDataset()

    # For each known operation, create samples with varying children counts
    for op, cost in OPERATION_COSTS.items():
        for num_children in [0, 1, 2, 3, 5]:
            # Determine type based on operation
            if "i32" in op:
                ty = "i32"
            elif "f64" in op:
                ty = "f64"
            elif "bool" in op:
                ty = "bool"
            else:
                ty = "Unit"

            dataset.add(
                op=op,
                ty=ty,
                num_children=num_children,
                predicted_cost=cost,
            )

    return dataset


class CostPredictor:
    """
    ML-based cost predictor with training and inference.

    Supports multiple backends:
    - 'linear': Simple linear regression
    - 'xgboost': Gradient boosted trees (if available)
    - 'mlp': Multi-layer perceptron (if sklearn available)
    """

    def __init__(self, backend: str = "linear"):
        self.backend = backend
        self.model: Any = None
        self._feature_names = NodeFeatures.feature_names()
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CostPredictor":
        """Train the model on features X and targets y."""
        if self.backend == "linear":
            self.model = self._fit_linear(X, y)
        elif self.backend == "xgboost":
            self.model = self._fit_xgboost(X, y)
        elif self.backend == "mlp":
            self.model = self._fit_mlp(X, y)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict costs for features X."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.backend == "linear":
            return self._predict_linear(X)
        elif self.backend == "xgboost":
            return self.model.predict(X)
        elif self.backend == "mlp":
            return self.model.predict(X)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def predict_single(self, features: list[float]) -> float:
        """Predict cost for a single feature vector."""
        X = np.array([features])
        return float(self.predict(X)[0])

    def _fit_linear(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Fit a simple linear model using normal equations."""
        # Add bias term
        X_bias = np.column_stack([np.ones(len(X)), X])

        # Solve normal equations: (X'X)^-1 X'y
        try:
            XtX = X_bias.T @ X_bias
            Xty = X_bias.T @ y
            # Add small regularization for numerical stability
            reg = 1e-6 * np.eye(XtX.shape[0])
            weights = np.linalg.solve(XtX + reg, Xty)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            weights = np.linalg.lstsq(X_bias, y, rcond=None)[0]

        return {"weights": weights}

    def _predict_linear(self, X: np.ndarray) -> np.ndarray:
        """Predict using linear model."""
        X_bias = np.column_stack([np.ones(len(X)), X])
        weights = self.model["weights"]
        return X_bias @ weights

    def _fit_xgboost(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Fit XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Use: pip install xgboost")

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="reg:squarederror",
        )
        model.fit(X, y)
        return model

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Fit MLP model."""
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scikit-learn not installed. Use: pip install scikit-learn")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_scaled, y)

        return {"mlp": model, "scaler": scaler}

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance (if supported by backend)."""
        if not self._is_fitted:
            return None

        if self.backend == "linear":
            weights = self.model["weights"][1:]  # Skip bias
            importance = dict(zip(self._feature_names, np.abs(weights)))
            return importance
        elif self.backend == "xgboost":
            importance = self.model.feature_importances_
            return dict(zip(self._feature_names, importance))
        else:
            return None

    def save(self, path: str | Path) -> None:
        """Save model to file."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "backend": self.backend,
                "model": self.model,
                "is_fitted": self._is_fitted,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "CostPredictor":
        """Load model from file."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        predictor = cls(backend=data["backend"])
        predictor.model = data["model"]
        predictor._is_fitted = data["is_fitted"]
        return predictor


def evaluate_model(
    predictor: CostPredictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate model performance.

    Returns:
        Dict with metrics: mae, rmse, r2, spearman_correlation
    """
    from scipy.stats import spearmanr

    y_pred = predictor.predict(X_test)

    # Mean Absolute Error
    mae = np.mean(np.abs(y_pred - y_test))

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

    # R-squared
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Spearman correlation (ranking quality)
    spearman, _ = spearmanr(y_pred, y_test)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "spearman_correlation": float(spearman) if np.isfinite(spearman) else 0.0,
    }


def train_cost_model(
    dataset: TrainingDataset,
    backend: str = "linear",
    test_split: float = 0.2,
) -> tuple[CostPredictor, dict[str, float]]:
    """
    Train a cost prediction model.

    Args:
        dataset: Training dataset
        backend: Model backend ('linear', 'xgboost', 'mlp')
        test_split: Fraction of data for testing

    Returns:
        Tuple of (trained predictor, evaluation metrics)
    """
    X, y = dataset.to_arrays()

    # Split data
    n_test = int(len(X) * test_split)
    if n_test < 1:
        n_test = 1

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    # Train
    predictor = CostPredictor(backend=backend)
    predictor.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(predictor, X_test, y_test)

    return predictor, metrics


class TrainedMLCostModel:
    """
    Cost model that uses a trained ML predictor.

    This wraps CostPredictor to provide the CostModel interface
    expected by the extraction algorithm.
    """

    def __init__(self, predictor: CostPredictor):
        self.predictor = predictor

    def get_cost_function(
        self,
        nodename: str,
        op: str,
        ty: str,
        cost: float,
        children: Sequence[str],
    ):
        """Return cost function for extraction."""
        from sealir.eqsat.rvsdg_extract import CostModel as _CostModel

        features = extract_features(op, ty, len(children))
        predicted = self.predictor.predict_single(features.to_vector())

        # Ensure non-negative
        predicted = max(0.1, predicted)

        # Create a simple cost function
        return _simple_cost_func(predicted)


def _simple_cost_func(cost: float):
    """Create a simple cost function that returns a constant."""
    from sealir.eqsat.rvsdg_extract import CostFunc
    return CostFunc(lambda *args, **kwargs: cost, {})


def create_pretrained_model() -> CostPredictor:
    """
    Create a pre-trained model using analytical costs.

    This provides a reasonable starting point without any
    measured runtime data.
    """
    # Generate training data from known costs
    dataset = generate_training_data_from_operations()

    # Train linear model (fast, interpretable)
    predictor, metrics = train_cost_model(dataset, backend="linear")

    print(f"Pre-trained model metrics:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Spearman: {metrics['spearman_correlation']:.4f}")

    return predictor
