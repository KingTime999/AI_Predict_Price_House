"""
Generate expanded training dataset from existing demo data patterns.
Creates 1000+ records with realistic variations to reduce overfitting.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load existing demo data
demo_file = "backend/data/demo_training_dataset.csv"
df = pd.read_csv(demo_file)

print(f"📖 Original demo data: {len(df)} rows")
print(f"   Columns: {df.columns.tolist()}\n")

# Set random seed for reproducibility
np.random.seed(42)

# Extract locations, directions, and other categories
locations = df['Address'].unique().tolist()
house_directions = df['House direction'].unique().tolist()
balcony_directions = df['Balcony direction'].unique().tolist()
legal_statuses = df['Legal status'].unique().tolist()
furniture_states = df['Furniture state'].unique().tolist()

print(f"📊 Categories found:")
print(f"   Locations: {len(locations)}")
print(f"   House directions: {len(house_directions)}")
print(f"   Legal statuses: {len(legal_statuses)}\n")

# Generate synthetic data
expanded_data = []

# Repeat and perturb existing data
for _ in range(10):  # 10x expansion = 1000 records
    for idx, row in df.iterrows():
        new_row = row.copy()
        
        # Add realistic noise to numerical features (±5-15%)
        numeric_cols = ['Frontage', 'Access Road', 'Area', 'Bedrooms', 'Bathrooms', 'Floors']
        for col in numeric_cols:
            if pd.notna(new_row[col]):
                noise = np.random.normal(1.0, 0.08)  # ±8% noise
                new_row[col] = max(0.5, new_row[col] * noise)
        
        # Add noise to price (±10-20%)
        if pd.notna(new_row['Price']):
            price_noise = np.random.normal(1.0, 0.12)  # ±12% noise
            new_row['Price'] = max(1000, new_row['Price'] * price_noise)
        
        # Randomly change some categorical features
        if np.random.rand() < 0.3:  # 30% chance to change
            new_row['House direction'] = np.random.choice(house_directions)
        if np.random.rand() < 0.2:
            new_row['Balcony direction'] = np.random.choice(balcony_directions)
        if np.random.rand() < 0.1:
            new_row['Legal status'] = np.random.choice(legal_statuses)
        if np.random.rand() < 0.15:
            new_row['Furniture state'] = np.random.choice(furniture_states)
        
        expanded_data.append(new_row)

# Create expanded DataFrame
expanded_df = pd.DataFrame(expanded_data)

# Reset index
expanded_df = expanded_df.reset_index(drop=True)

# Save to file
output_file = "backend/data/demo_training_dataset.csv"
expanded_df.to_csv(output_file, index=False)

print(f"✅ Generated expanded dataset: {len(expanded_df)} rows")
print(f"   Saved to: {output_file}\n")

# Show statistics
print("📈 Dataset statistics:")
print(f"   Price range: {expanded_df['Price'].min():.2f} - {expanded_df['Price'].max():.2f} million VND")
print(f"   Area range: {expanded_df['Area'].min():.1f} - {expanded_df['Area'].max():.1f} m²")
print(f"   Bedrooms: {expanded_df['Bedrooms'].min():.0f} - {expanded_df['Bedrooms'].max():.0f}")
print(f"   Unique locations: {expanded_df['Address'].nunique()}")
