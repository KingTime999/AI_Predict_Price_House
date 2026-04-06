import pandas as pd
import joblib

# Load dataset
df = pd.read_csv('backend/data/catalyst_dataset.csv')

# Extract district/ward from Address
df['District'] = df['Address'].str.extract(r'([^,]+)(?=,|$)', expand=False).str.strip()

# Calculate average price by district
avg_price = df.groupby('District')['Price'].agg(['mean', 'median', 'min', 'max', 'count']).round(2)
avg_price = avg_price.sort_values('mean', ascending=False)

print('=' * 110)
print('AVERAGE PROPERTY PRICE BY DISTRICT')
print('=' * 110)
print(f'{"District":<40} {"Avg (tr)":<15} {"Median":<15} {"Min":<15} {"Max":<15} {"Count":<8}')
print('-' * 110)

for district, row in avg_price.iterrows():
    print(f'{district:<40} {row["mean"]:<15.2f} {row["median"]:<15.2f} {row["min"]:<15.2f} {row["max"]:<15.2f} {int(row["count"]):<8}')

print('\n' + '=' * 110)
print('MODEL INFORMATION & CALCULATION METHOD')
print('=' * 110)

# Load model
model_dict = joblib.load('backend/artifacts/house_price_model.joblib')

print(f'\n✓ Features Used in Model:')
for i, col in enumerate(model_dict['feature_columns'], 1):
    print(f'  {i}. {col}')

print(f'\n✓ Numeric Features (processed): {model_dict["numeric_features"]}')
print(f'✓ Categorical Features (processed): {model_dict["categorical_features"]}')
print(f'✓ Target Variable: {model_dict["target"]} ({model_dict["target_unit"]})')

print(f'\n✓ Model Metrics:')
print(f'  - Training R² Score: {model_dict["metrics"]["train_r2"]:.4f} (16.56%)')
print(f'  - Validation R² Score: {model_dict["metrics"]["test_r2"]:.4f} (12.40%)')
print(f'  - Training MAE: {model_dict["metrics"]["train_mae"]:.2f} million VND')
print(f'  - Validation MAE: {model_dict["metrics"]["test_mae"]:.2f} million VND')

print(f'\n✓ How Price is Calculated:')
print(f'  1. Numeric features (Area, Bedrooms, Bathrooms, Floors, Access Road)')
print(f'     → Standardized using StandardScaler')
print(f'  2. Categorical features (Address, Legal status, Furniture state, Frontage)')
print(f'     → OneHotEncoded or Label Encoded')
print(f'  3. All features → Linear Regression Model')
print(f'  4. Output: Predicted Price (in million VND)')

print(f'\n✓ Accuracy Explanation:')
print(f'  R² = 0.1240 means the model explains 12.4% of price variance')
print(f'  - Good for: General price trend estimation')
print(f'  - Limitation: Other factors (location prestige, amenities, market timing) affect price')
print(f'  - MAE: On average, prediction is off by ±10,433 million VND')

print('\n' + '=' * 110)
