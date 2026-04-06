import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.stdout.reconfigure(encoding='utf-8')

# Get dataset path
data_dir = Path(__file__).parent.parent / 'data'
dataset_path = data_dir / 'catalyst_dataset.csv'

print("=" * 80)
print("RETRAIN MODEL - Advanced Feature Engineering")
print("=" * 80)

# Load dataset
print(f"\n📊 Loading dataset from {dataset_path}")
df = pd.read_csv(dataset_path)
print(f"   Total records: {len(df):,}")

# Extract District from Address
print("\n🏘️ Extracting District from Address...")
df['District'] = df['Address'].str.extract(r'([^,]+)', expand=False).str.strip()
print(f"   Unique districts: {df['District'].nunique()}")

# Remove obvious outliers (price=0 or extreme outliers)
print("\n🔍 Removing outliers...")
# Define reasonable price range: 500M - 200,000M VND (based on typical Vietnam prices)
df_clean = df[(df['Price'] > 500) & (df['Price'] < 200000)].copy()
print(f"   Records after outlier removal: {len(df_clean):,}")
print(f"   Price range: {df_clean['Price'].min():.0f} - {df_clean['Price'].max():.0f} tr VND")

# Fill missing numeric values
print("\n📈 Filling missing values...")
df_clean['Bedrooms'] = df_clean['Bedrooms'].fillna(df_clean['Bedrooms'].median())
df_clean['Bathrooms'] = df_clean['Bathrooms'].fillna(df_clean['Bathrooms'].median())
df_clean['Floors'] = df_clean['Floors'].fillna(df_clean['Floors'].median())
print(f"   Bedrooms: {df_clean['Bedrooms'].isna().sum()} na")
print(f"   Bathrooms: {df_clean['Bathrooms'].isna().sum()} na")
print(f"   Floors: {df_clean['Floors'].isna().sum()} na")

# Feature engineering
print("\n⚙️ Building features...")
X = df_clean[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Frontage', 'District']].copy()
y = df_clean['Price'].copy()

print(f"   Initial shape: {X.shape}")

# Encode Frontage (boolean to int)
X['Frontage'] = X['Frontage'].astype(int)

# Encode District (ALL districts, no grouping)
print(f"   Districts before encoding: {X['District'].nunique()}")

# One-hot encode District
X = pd.get_dummies(X, columns=['District'], prefix='District', drop_first=True)
print(f"   Districts after one-hot encoding: {len([c for c in X.columns if c.startswith('District_')])}")
print(f"   Final feature shape after one-hot encoding: {X.shape}")
print(f"   Features: {len(X.columns)}")

# Split data (80% train, 20% test)
print("\n📊 Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training samples: {len(X_train):,}")
print(f"   Testing samples: {len(X_test):,}")

# Train RandomForestRegressor
print("\n🤖 Training RandomForestRegressor (n_estimators=100)...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
model.fit(X_train, y_train)

# Evaluate
print("\n📈 Evaluating model...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n{'Metric':<20} {'Train':<20} {'Test':<20}")
print("=" * 60)
print(f"{'R² Score':<20} {train_r2:.4f} ({train_r2*100:.2f}%) {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"{'MAE (tr VND)':<20} {train_mae:,.0f} {test_mae:,.0f}")
print(f"{'RMSE (tr VND)':<20} {train_rmse:,.0f} {test_rmse:,.0f}")

# Feature importance
print("\n🎯 Top 10 Important Features:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, (feat, imp) in enumerate(zip(feature_importance.head(10)['Feature'], 
                                       feature_importance.head(10)['Importance']), 1):
    print(f"   {idx:2d}. {feat:<30} {imp:.4f}")

# Save model
model_path = Path(__file__).parent.parent / 'artifacts' / 'house_price_model.joblib'
model_path.parent.mkdir(exist_ok=True)
print(f"\n💾 Saving model to {model_path}")

# Store both model and feature columns for later use
model_data = {
    'model': model,
    'feature_columns': X.columns.tolist(),
    'r2_score': test_r2,
}
joblib.dump(model_data, model_path)

print("\n✅ Model retrained and saved successfully!")
print(f"   Accuracy (R² on test set): {test_r2*100:.2f}%")
print(f"   Average error (MAE): {test_mae:,.0f} tr VND")
