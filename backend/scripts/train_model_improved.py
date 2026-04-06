import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sys

# Fix encoding for Vietnamese characters
sys.stdout.reconfigure(encoding='utf-8')

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

def extract_district(address: str) -> str:
    """Extract district from address (first part before comma)"""
    if pd.isna(address):
        return "Unknown"
    parts = str(address).split(',')
    if len(parts) > 0:
        return parts[0].strip()
    return "Unknown"

def remove_outliers(df: pd.DataFrame, column: str, lower_percentile: float = 1, upper_percentile: float = 99) -> pd.DataFrame:
    """Remove price outliers using percentile method"""
    lower = df[column].quantile(lower_percentile / 100)
    upper = df[column].quantile(upper_percentile / 100)
    print(f"  Removing prices outside range: {lower:.2f} - {upper:.2f} million VND")
    return df[(df[column] >= lower) & (df[column] <= upper)]

def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    # Parse numeric columns
    numeric_cols = ["Frontage", "Access Road", "Area", "Bedrooms", "Bathrooms", "Floors", "Price"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_number)

    # Convert Frontage to binary
    if "Frontage" in df.columns:
        df["Frontage"] = (df["Frontage"] > 0).astype(int)

    # Fill missing values by Address
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

    # Extract district from Address
    if "Address" in df.columns:
        df["District"] = df["Address"].apply(extract_district)

    return df

def train(csv_path: str, output_path: str) -> dict:
    print(f"Loading dataset from {csv_path}...")
    frame = pd.read_csv(csv_path)
    print(f"  Loaded {len(frame)} records")

    print(f"\nPreprocessing data...")
    df = preprocess(frame)

    # Define features - now including District
    num_features = ["Frontage", "Access Road", "Area", "Bedrooms", "Bathrooms", "Floors"]
    cat_features = ["District", "Legal status", "Furniture state"]
    feature_cols = num_features + cat_features

    # Check required columns
    missing = [col for col in feature_cols + ["Price"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Prepare target
    y = df["Price"].apply(parse_number)
    valid_mask = y.notna() & (y > 0)
    
    X = df.loc[valid_mask, feature_cols]
    y = y.loc[valid_mask].astype(float)

    print(f"  Valid records (Price > 0): {len(y)}")

    # Remove outliers
    print(f"\nRemoving outliers...")
    valid_data = pd.concat([X, y], axis=1)
    valid_data = remove_outliers(valid_data, "Price", lower_percentile=1, upper_percentile=99)
    
    X = valid_data[feature_cols]
    y = valid_data["Price"]
    
    print(f"  Records after outlier removal: {len(y)}")
    print(f"  Price range: {y.min():.2f} - {y.max():.2f} million VND")
    print(f"  Average price: {y.mean():.2f} million VND")
    print(f"  Distinct districts: {X['District'].nunique()}")

    # Create preprocessing pipelines
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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # Split data
    print(f"\nTraining model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Calculate metrics
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)

    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    print(f"\n{'='*70}")
    print(f"Training Metrics:")
    print(f"  MAE:  {train_mae:,.2f} million VND")
    print(f"  RMSE: {train_rmse:,.2f} million VND")
    print(f"  R²:   {train_r2:.4f} ({train_r2*100:.2f}%)")
    
    print(f"\nValidation Metrics:")
    print(f"  MAE:  {test_mae:,.2f} million VND")
    print(f"  RMSE: {test_rmse:,.2f} million VND")
    print(f"  R²:   {test_r2:.4f} ({test_r2*100:.2f}%)")
    print(f"{'='*70}")

    # Sample predictions
    print(f"\nSample Predictions on Test Set:")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    for idx in sample_indices:
        actual = y_test.iloc[idx]
        predicted = test_pred[idx]
        error = abs(actual - predicted)
        error_pct = (error / actual * 100) if actual > 0 else 0
        district = X_test.iloc[idx]["District"]
        print(f"  {district}: Actual={actual:,.0f}, Predicted={predicted:,.0f}, Error={error_pct:.1f}%")

    # Save artifact
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
        "data_info": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "price_range": [float(y.min()), float(y.max())],
            "avg_price": float(y.mean()),
        }
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    print(f"\n✓ Model saved to: {output_path}")
    return artifact["metrics"]

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Improved model training with district-based features"
    )
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument(
        "--output",
        default="backend/artifacts/house_price_model.joblib",
        help="Model artifact output path",
    )
    args = parser.parse_args()
    train(args.csv, args.output)

if __name__ == "__main__":
    main()
