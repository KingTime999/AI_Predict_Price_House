import pandas as pd
import sys
import os

# Fix encoding for Vietnamese characters
sys.stdout.reconfigure(encoding='utf-8')

# Get the data directory path
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', 'data')
dataset_path = os.path.join(data_dir, 'catalyst_dataset.csv')
output_path = os.path.join(data_dir, 'price_by_district.csv')

# Load dataset
df = pd.read_csv(dataset_path)

# Extract district from Address (first part before comma)
df['District'] = df['Address'].str.extract(r'([^,]+)', expand=False).str.strip()

# Calculate statistics by district
stats = df.groupby('District')['Price'].agg([
    ('Count', 'count'),
    ('Avg_Price', 'mean'),
    ('Min_Price', 'min'),
    ('Max_Price', 'max'),
    ('Median_Price', 'median')
]).round(2)

# Sort by average price descending
stats = stats.sort_values('Avg_Price', ascending=False)

# Reset index to become a column
stats = stats.reset_index()
stats.columns = ['District', 'Count', 'Avg_Price', 'Min_Price', 'Max_Price', 'Median_Price']

# Save to CSV
stats.to_csv(output_path, index=False, encoding='utf-8')

print(f'✓ File saved: {output_path}')
print(f'\nSummary:')
print(f'  Total districts: {len(stats)}')
print(f'  Total properties: {len(df)}')
print(f'  Average price nationwide: {df["Price"].mean():.2f} million VND')
print(f'  Median price nationwide: {df["Price"].median():.2f} million VND')
print(f'  Price range: {df["Price"].min():.2f} - {df["Price"].max():.2f} million VND')

# Display first 10 rows
print(f'\n--- Top 10 Districts by Average Price ---')
print(stats.head(10).to_string(index=False))
