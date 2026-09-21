"""Train, evaluate, and save the best salary prediction model."""
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from data_loader import load_data
from feature_engineering import get_feature_columns, build_preprocessor

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    # 1. Load data
    print("Loading data...")
    df = load_data()
    print(f"  Shape: {df.shape}")

    # 2. Features & target
    cat_cols, num_cols = get_feature_columns(df)
    print(f"  Categorical: {cat_cols}")
    print(f"  Numerical: {num_cols}")

    X = df.drop("salary", axis=1)
    y = df["salary"]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 4. Preprocessor
    preprocessor = build_preprocessor(cat_cols, num_cols)

    # 5. Train & compare models
    models = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    }

    results = {}
    best_name, best_r2 = None, -1
    best_pipe = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
        print(f"  R²={r2:.4f} | MAE=${mae:,.0f} | RMSE=${rmse:,.0f}")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipe

    # 6. Save the best model
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_pipe, model_path)
    print(f"\n{'='*50}")
    print(f"Best model: {best_name} (R²={best_r2:.4f})")
    print(f"Saved to: {model_path}")

    # 7. Save metrics
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
