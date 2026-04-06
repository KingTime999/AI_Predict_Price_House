import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.stdout.reconfigure(encoding='utf-8')

# Get dataset path
data_dir = Path(__file__).parent.parent / 'data'
dataset_path = data_dir / 'catalyst_dataset.csv'

print("=" * 80)
print("ADVANCED TRAINING - All Districts with Optimization")
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
df_clean = df[(df['Price'] > 500) & (df['Price'] < 200000)].copy()
print(f"   Records after outlier removal: {len(df_clean):,}")

# Fill missing numeric values
print("\n📈 Filling missing values...")
df_clean['Bedrooms'] = df_clean['Bedrooms'].fillna(df_clean['Bedrooms'].median())
df_clean['Bathrooms'] = df_clean['Bathrooms'].fillna(df_clean['Bathrooms'].median())
df_clean['Floors'] = df_clean['Floors'].fillna(df_clean['Floors'].median())

# Feature engineering
print("\n⚙️ Building features with engineering...")
X = df_clean[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Frontage', 'District']].copy()
y = df_clean['Price'].copy()

print(f"   Initial shape: {X.shape}")

# Encode Frontage (boolean to int)
X['Frontage'] = X['Frontage'].astype(int)

# Encode District (ALL districts)
print(f"   Districts: {X['District'].nunique()}")
X = pd.get_dummies(X, columns=['District'], prefix='District', drop_first=True)

# Add interaction features (Area x Floors, Area x Bedrooms)
print(f"   Adding interaction features...")
X['Area_Floors'] = X['Area'] * X['Floors']
X['Area_Bedrooms'] = X['Area'] * X['Bedrooms']
X['Bedrooms_Bathrooms'] = X['Bedrooms'] * X['Bathrooms']

print(f"   Final feature shape: {X.shape}")
print(f"   Total features: {len(X.columns)}")

# Split data
print("\n📊 Splitting data (80/20) with stratification...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training samples: {len(X_train):,}")
print(f"   Testing samples: {len(X_test):,}")

# Train improved RandomForest
print("\n🤖 Training Improved RandomForestRegressor (n_estimators=200)...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
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

# Cross-validation
print("\n🔍 Cross-validation (5-fold)...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
print(f"   CV R² Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(f"\n{'Metric':<20} {'Train':<20} {'Test':<20}")
print("=" * 60)
print(f"{'R² Score':<20} {train_r2:.4f} ({train_r2*100:.2f}%) {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"{'MAE (tr VND)':<20} {train_mae:,.0f} {test_mae:,.0f}")
print(f"{'RMSE (tr VND)':<20} {train_rmse:,.0f} {test_rmse:,.0f}")

# Feature importance
print("\n🎯 Top 15 Important Features:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, (feat, imp) in enumerate(zip(feature_importance.head(15)['Feature'], 
                                       feature_importance.head(15)['Importance']), 1):
    print(f"   {idx:2d}. {feat:<35} {imp:.4f}")

# Save model
model_path = Path(__file__).parent.parent / 'artifacts' / 'house_price_model.joblib'
model_path.parent.mkdir(exist_ok=True)

print(f"\n💾 Saving model to {model_path}")
model_data = {
    'model': model,
    'feature_columns': X.columns.tolist(),
    'r2_score': test_r2,
    'cv_mean': cv_scores.mean(),
}
joblib.dump(model_data, model_path)

print("\n✅ Model trained and saved successfully!")
print(f"   Test R² (Accuracy): {test_r2*100:.2f}%")
print(f"   CV R² (Mean): {cv_scores.mean()*100:.2f}%")
print(f"   Average error (MAE): {test_mae:,.0f} tr VND")
print(f"   Improvement vs previous: +{(test_r2 - 0.6288)*100:.2f}%")
