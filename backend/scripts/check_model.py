import joblib

m = joblib.load('backend/artifacts/house_price_model.joblib')
print(f'R² score stored: {m.get("r2_score", 0):.4f}')
print(f'Total features: {len(m["feature_columns"])}')
