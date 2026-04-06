import pandas as pd

df = pd.read_csv('catalyst_dataset.csv')

# Filter realistic values
df_clean = df[(df['Area'] > 0) & (df['Area'] < 500) & (df['Price'] > 0) & (df['Price'] < 100000)]

print(f"Total records: {len(df)}")
print(f"After removing outliers: {len(df_clean)}")
print(f"Area range (clean): {df_clean['Area'].min():.0f} - {df_clean['Area'].max():.0f} m²")
print(f"Area median: {df_clean['Area'].median():.0f} m²")
print(f"Price range (clean): {df_clean['Price'].min():.0f} - {df_clean['Price'].max():.0f} tr VND")
print(f"Price median: {df_clean['Price'].median():.0f} tr VND")
print(f"Locations (unique): {df['Address'].nunique()}")
print(f"\nTop 10 locations:")
print(df['Address'].value_counts().head(10))
