import argparse
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

    # Convert Frontage to binary (1 if has frontage, 0 otherwise)
    if "Frontage" in df.columns:
        df["Frontage"] = (df["Frontage"] > 0).astype(int)

    if "Address" in df.columns and "Access Road" in df.columns:
        df["Access Road"] = df.groupby("Address")["Access Road"].transform(fill_mode)

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


def train(csv_path: str, output_path: str) -> dict[str, float]:
    frame = pd.read_csv(csv_path)
    df = preprocess(frame)

    num_features = ["Frontage", "Access Road", "Area", "Bedrooms", "Bathrooms", "Floors"]
    cat_features = ["Address", "Legal status", "Furniture state"]
    feature_cols = num_features + cat_features

    missing = [col for col in feature_cols + ["Price"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    y = df["Price"].apply(parse_number)
    valid_mask = y.notna()

    X = df.loc[valid_mask, feature_cols]
    y = y.loc[valid_mask].astype(float)

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model.fit(X_train, y_train)
    
    # Calculate training metrics
    train_predictions = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    train_r2 = model.score(X_train, y_train)
    
    # Calculate test metrics
    test_predictions = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_r2 = model.score(X_test, y_test)

    artifact = {
        "pipeline": model,
        "metrics": {
            "train_mae": float(train_mae),
            "train_rmse": float(train_rmse),
            "train_r2": float(train_r2),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
        },
        "feature_columns": feature_cols,
        "numeric_features": num_features,
        "categorical_features": cat_features,
        "target": "Price",
        "target_unit": "million_vnd",
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    return artifact["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Linear Regression model for Vietnam house prices")
    parser.add_argument("--csv", required=True, help="Path to vietnam_housing_dataset.csv")
    parser.add_argument(
        "--output",
        default="backend/artifacts/house_price_model.joblib",
        help="Model artifact output path",
    )
    args = parser.parse_args()

    metrics = train(args.csv, args.output)

    print(f"\n{'='*60}")
    print(f"Model trained and saved to {args.output}")
    print(f"{'='*60}")
    print(f"\n📊 TRAINING SET METRICS:")
    print(f"  • MAE:  {metrics['train_mae']:.4f} million VND")
    print(f"  • RMSE: {metrics['train_rmse']:.4f} million VND")
    print(f"  • R² Score: {metrics['train_r2']:.4f}")
    
    print(f"\n📊 TEST SET METRICS (Validation):")
    print(f"  • MAE:  {metrics['test_mae']:.4f} million VND")
    print(f"  • RMSE: {metrics['test_rmse']:.4f} million VND")
    print(f"  • R² Score: {metrics['test_r2']:.4f}")
    
    print(f"\n🔍 OVERFITTING ANALYSIS:")
    mae_gap = metrics['test_mae'] - metrics['train_mae']
    rmse_gap = metrics['test_rmse'] - metrics['train_rmse']
    r2_gap = metrics['train_r2'] - metrics['test_r2']
    
    print(f"  • MAE Gap (Test - Train): {mae_gap:+.4f} (higher = worse)")
    print(f"  • RMSE Gap (Test - Train): {rmse_gap:+.4f} (higher = worse)")
    print(f"  • R² Gap (Train - Test): {r2_gap:+.4f} (higher = overfitting)")
    
    if r2_gap > 0.1:
        print(f"\n⚠️  OVERFITTING DETECTED! R² gap > 0.1")
    elif r2_gap > 0.05:
        print(f"\n⚡ Moderate overfitting detected (0.05 < R² gap < 0.1)")
    else:
        print(f"\n✅ Model is well-balanced (minimal overfitting)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
