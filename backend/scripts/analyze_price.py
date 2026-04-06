import pandas as pd
import sys

# Fix encoding for Vietnamese characters
sys.stdout.reconfigure(encoding='utf-8')

# Load dataset
df = pd.read_csv('catalyst_dataset.csv')

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

# Print results
print('='*100)
print('PHÂN TÍCH GIÁ BĐS THEO QUẬN/PHƯỜNG (TẤT CẢ 261 QUẬN)')
print('='*100)
print(f'{"Quận/Phường":<40} {"Số BĐS":>10} {"Giá TB":>15} {"Giá Min":>15} {"Giá Max":>15} {"Giá Median":>15}')
print('-'*100)

for idx, (district, row) in enumerate(stats.iterrows(), 1):
    print(f'{district:<40} {int(row["Count"]):>10} {row["Avg_Price"]:>15.2f} {row["Min_Price"]:>15.2f} {row["Max_Price"]:>15.2f} {row["Median_Price"]:>15.2f}')

print('='*100)
print(f'\nTổng cộng: {len(df)} BĐS tại {len(stats)} quận/phường')
print(f'Giá trung bình toàn thị trường: {df["Price"].mean():.2f} triệu VND')
print(f'Giá trung vị toàn thị trường: {df["Price"].median():.2f} triệu VND')
print(f'Khoảng giá: {df["Price"].min():.2f} - {df["Price"].max():.2f} triệu VND')
