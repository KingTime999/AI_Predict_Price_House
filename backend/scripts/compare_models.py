"""
Compare different ML models to find the best one with minimal overfitting.
"""
import argparse
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    normalized = (
        text.replace("m2", "")
        .replace("m^2", "")
        .replace("m", "")
        .replace(" ", "")
    )
    normalized = normalized.replace(",", ".")

    matches = re.findall(r"-?\d+(?:\.\d+)?", normalized)
    if not matches:
        return None

    try:
        first = float(matches[0])
        has_ty = "tỷ" in lowered or "ty" in lowered or "billion" in lowered
        has_trieu = "triệu" in lowered or "trieu" in lowered or "million" in lowered

        if has_ty and has_trieu and len(matches) >= 2:
            second = float(matches[1])
            return first * 1000 + second

        if has_ty:
            return first * 1000

        return first
    except ValueError:
        return None


def fill_mode(series: pd.Series) -> pd.Series:
    mode = series.mode(dropna=True)
    if not mode.empty:
        return series.fillna(mode.iloc[0])
    return series


def fill_median(series: pd.Series) -> pd.Series:
    median = series.median(skipna=True)
    if pd.notna(median):
        return series.fillna(median)
    return series


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    numeric_cols = ["Frontage", "Access Road", "Area", "Bedrooms", "Bathrooms", "Floors", "Price"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_number)

    if "Address" in df.columns and "Frontage" in df.columns:
        df["Frontage"] = df.groupby("Address")["Frontage"].transform(fill_mode)
    if "Address" in df.columns and "Access Road" in df.columns:
        df["Access Road"] = df.groupby("Address")["Access Road"].transform(fill_mode)

    if "House direction" in df.columns:
        df["House direction"] = df["House direction"].fillna("N/A")
    if "Balcony direction" in df.columns:
        df["Balcony direction"] = df["Balcony direction"].fillna("N/A")

    if "Address" in df.columns and "Floors" in df.columns:
        df["Floors"] = df.groupby("Address")["Floors"].transform(fill_mode)
    if "Floors" in df.columns:
        df["Floors"] = df["Floors"].fillna(1)

    if "Address" in df.columns and "Area" in df.columns:
        df["Area"] = df.groupby("Address")["Area"].transform(fill_median)
    if "Area" in df.columns:
        df["Area"] = df["Area"].fillna(df["Area"].median())

    if "Address" in df.columns and "Bedrooms" in df.columns:
        df["Bedrooms"] = df.groupby("Address")["Bedrooms"].transform(fill_median)
    if "Address" in df.columns and "Bathrooms" in df.columns:
        df["Bathrooms"] = df.groupby("Address")["Bathrooms"].transform(fill_median)

    if "Bedrooms" in df.columns:
        df["Bedrooms"] = df["Bedrooms"].fillna(1)
    if "Bathrooms" in df.columns:
        df["Bathrooms"] = df["Bathrooms"].fillna(1)

    if "Legal status" in df.columns:
        df["Legal status"] = df["Legal status"].fillna("Sale contract")
    if "Furniture state" in df.columns:
        df["Furniture state"] = df["Furniture state"].fillna("N/A")

    return df


def create_preprocessor(num_features, cat_features):
    """Create a preprocessing pipeline."""
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )
    return preprocessor


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model, return metrics."""
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    r2_gap = train_r2 - test_r2
    mae_gap = test_mae - train_mae
    rmse_gap = test_rmse - train_rmse
    
    return {
        "name": name,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "r2_gap": r2_gap,
        "mae_gap": mae_gap,
        "rmse_gap": rmse_gap,
    }


def compare_models(csv_path):
    """Compare multiple models on the same dataset."""
    frame = pd.read_csv(csv_path)
    df = preprocess(frame)

    num_features = ["Frontage", "Access Road", "Area", "Bedrooms", "Bathrooms", "Floors"]
    cat_features = ["Address", "House direction", "Balcony direction", "Legal status", "Furniture state"]
    feature_cols = num_features + cat_features

    y = df["Price"].apply(parse_number)
    valid_mask = y.notna()

    X = df.loc[valid_mask, feature_cols]
    y = y.loc[valid_mask].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = create_preprocessor(num_features, cat_features)

    models = {
        "Linear Regression": Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())]),
        "Ridge (α=1.0)": Pipeline([("preprocessor", preprocessor), ("regressor", Ridge(alpha=1.0))]),
        "Ridge (α=10.0)": Pipeline([("preprocessor", preprocessor), ("regressor", Ridge(alpha=10.0))]),
        "Lasso (α=0.1)": Pipeline([("preprocessor", preprocessor), ("regressor", Lasso(alpha=0.1, max_iter=10000))]),
        "Random Forest": Pipeline([("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("preprocessor", preprocessor), ("regressor", GradientBoostingRegressor(n_estimators=100, random_state=42))]),
    }

    results = []
    for name, model in models.items():
        print(f"  ⏳ Training {name}...", end="", flush=True)
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        print(" ✓")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare different ML models for Vietnam house prices")
    parser.add_argument("--csv", required=True, help="Path to training CSV")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🔬 MODEL COMPARISON - Looking for least overfitting")
    print("="*80 + "\n")

    results = compare_models(args.csv)

    # Sort by R² gap (lower is better)
    results_sorted = sorted(results, key=lambda x: x["r2_gap"])

    print("\n" + "="*80)
    print("📊 RESULTS (sorted by R² Gap - lower = less overfitting):")
    print("="*80 + "\n")

    for i, metrics in enumerate(results_sorted, 1):
        name = metrics["name"]
        r2_gap = metrics["r2_gap"]
        test_r2 = metrics["test_r2"]
        test_mae = metrics["test_mae"]
        test_rmse = metrics["test_rmse"]

        # Overfitting severity
        if r2_gap > 0.2:
            severity = "🔴 SEVERE"
        elif r2_gap > 0.1:
            severity = "🟠 HIGH"
        elif r2_gap > 0.05:
            severity = "🟡 MODERATE"
        else:
            severity = "🟢 LOW"

        print(f"{i}. {name}")
        print(f"   ├─ Test R²: {test_r2:.4f}")
        print(f"   ├─ Test MAE: {test_mae:.2f} million VND")
        print(f"   ├─ Test RMSE: {test_rmse:.2f} million VND")
        print(f"   ├─ R² Gap (Train-Test): {r2_gap:.4f} {severity}")
        print(f"   └─ MAE Gap (Test-Train): {metrics['mae_gap']:+.2f} million VND\n")

    # Best model recommendation
    best = results_sorted[0]
    print("="*80)
    print(f"🏆 BEST MODEL (Least Overfitting): {best['name']}")
    print(f"   R² Gap: {best['r2_gap']:.4f}")
    print(f"   Test R²: {best['test_r2']:.4f}")
    print(f"   Test MAE: {best['test_mae']:.2f} million VND")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
