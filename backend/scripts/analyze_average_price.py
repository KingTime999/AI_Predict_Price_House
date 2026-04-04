#!/usr/bin/env python3
"""
Script to calculate average house prices by district/city
"""

import pandas as pd
from pathlib import Path

# Read CSV file
csv_path = Path(__file__).parent.parent / "data" / "demo_training_dataset.csv"
df = pd.read_csv(csv_path)

# Calculate average price by address (district/city)
avg_price_by_address = df.groupby('Address')['Price'].agg([
    ('Count', 'count'),
    ('Average_Price_Million_VND', 'mean'),
    ('Min_Price_Million_VND', 'min'),
    ('Max_Price_Million_VND', 'max'),
    ('Std_Dev', 'std')
]).round(2)

# Sort by average price in descending order
avg_price_by_address = avg_price_by_address.sort_values('Average_Price_Million_VND', ascending=False)

print("\n" + "="*100)
print("📊 PHÂN TÍCH GIÁ NHÀ TRUNG BÌNH THEO QUẬN/THÀNH PHỐ")
print("="*100)
print(f"\n✅ Tổng số căn nhà: {len(df)}")
print(f"✅ Số quận/thành phố: {len(avg_price_by_address)}")
print(f"✅ Giá trung bình chung: {df['Price'].mean():.2f} triệu VND\n")

print(avg_price_by_address.to_string())

print("\n" + "="*100)
print("📈 TOP 5 QUẬN CÓ GIÁ NHÀ CAO NHẤT")
print("="*100)
for idx, (address, row) in enumerate(avg_price_by_address.head(5).iterrows(), 1):
    print(f"{idx}. {address}")
    print(f"   💰 Trung bình: {row['Average_Price_Million_VND']:,.0f} triệu VND")
    print(f"   📍 Từ {row['Min_Price_Million_VND']:,.0f} đến {row['Max_Price_Million_VND']:,.0f} triệu VND")
    print(f"   🏠 Số căn nhà: {int(row['Count'])}\n")

print("\n" + "="*100)
print("📉 TOP 5 QUẬN CÓ GIÁ NHÀ THẤP NHẤT")
print("="*100)
for idx, (address, row) in enumerate(avg_price_by_address.tail(5).iterrows(), 1):
    print(f"{idx}. {address}")
    print(f"   💰 Trung bình: {row['Average_Price_Million_VND']:,.0f} triệu VND")
    print(f"   📍 Từ {row['Min_Price_Million_VND']:,.0f} đến {row['Max_Price_Million_VND']:,.0f} triệu VND")
    print(f"   🏠 Số căn nhà: {int(row['Count'])}\n")

# Save to CSV
output_csv = Path(__file__).parent.parent / "data" / "average_price_by_district.csv"
avg_price_by_address.to_csv(output_csv)
print(f"✅ Kết quả đã được lưu vào: {output_csv}")
print("="*100 + "\n")
