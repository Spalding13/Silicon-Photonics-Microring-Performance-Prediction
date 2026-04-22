"""Baseline regression models used in the notebook."""

from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BaseModel:
    """Shared fit / predict / evaluate helpers."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the wrapped model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference with the fitted model."""
        if not self.fitted:
            raise RuntimeError(f"{self.name} not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute RMSE, MAE, and R2 on a holdout set."""
        y_pred = self.predict(X)
        return {
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'mae': float(mean_absolute_error(y, y_pred)),
            'r2': float(r2_score(y, y_pred)),
        }


class LinearRegressionModel(BaseModel):
    """Plain linear regression."""

    def __init__(self):
        super().__init__("Linear Regression")
        self.model = LinearRegression()


class RidgeRegressionModel(BaseModel):
    """Ridge regression with optional group-aware alpha selection."""

    def __init__(self, alphas: Optional[np.ndarray] = None, cv_splits: int = 5):
        super().__init__("Ridge Regression")
        if alphas is None:
            alphas = np.logspace(-3, 3, 50)
        self.alphas = np.asarray(alphas, dtype=float)
        self.cv_splits = cv_splits
        self.selected_alpha_ = float(self.alphas[0])
        self.model = Ridge(alpha=self.selected_alpha_)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> None:
        """Fit Ridge after selecting alpha with cross-validation."""
        if groups is not None:
            n_unique_groups = len(np.unique(groups))
            n_splits = min(self.cv_splits, n_unique_groups)
            if n_splits < 2:
                raise ValueError("RidgeRegressionModel requires at least 2 groups for group-aware CV")
            cv = GroupKFold(n_splits=n_splits)
            cv_kwargs = {'groups': groups}
            cv_label = "group-aware"
        else:
            n_splits = min(self.cv_splits, len(X))
            if n_splits < 2:
                self.selected_alpha_ = float(self.alphas[0])
                self.model = Ridge(alpha=self.selected_alpha_)
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                self.fitted = True
                print(f"{self.name}: selected alpha={self.selected_alpha_:.6f} (single-split fallback)")
                return
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_kwargs = {}
            cv_label = "standard"

        best_alpha = float(self.alphas[0])
        best_score = -np.inf

        for alpha in self.alphas:
            pipeline = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring='neg_root_mean_squared_error',
                **cv_kwargs,
            )
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = float(alpha)

        self.selected_alpha_ = best_alpha
        self.model = Ridge(alpha=self.selected_alpha_)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True
        print(f"{self.name}: selected alpha={self.selected_alpha_:.6f} ({cv_label} CV)")


class HistGradientBoostingModel(BaseModel):
    """Simple nonlinear baseline for the tabular features."""

    def __init__(
        self,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        n_iter_no_change: int = 10,
    ):
        super().__init__("HistGradientBoosting")
        self.model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_iter_no_change=n_iter_no_change,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
        )


def get_all_baseline_models() -> Dict[str, BaseModel]:
    """Return the three baseline models used in the project."""
    return {
        'Linear': LinearRegressionModel(),
        'Ridge': RidgeRegressionModel(),
        'HistGBDT': HistGradientBoostingModel(),
    }
