import requests
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

districts = [
    ("Hoàn Kiếm", "Hoan Kiem, Ha Noi"),
    ("Ba Đình", "Ba Dinh, Ha Noi"),
    ("Hai Bà Trưng", "Hai Ba Trung, Ha Noi"),
    ("Đống Đa", "Dong Da, Ha Noi"),
    ("Cầu Giấy", "Cau Giay, Ha Noi"),
    ("Tây Hồ", "Tay Ho, Ha Noi"),
    ("Hà Đông", "Ha Dong, Ha Noi"),
    ("Long Biên", "Long Bien, Ha Noi"),
    ("Quận 1", "Quan 1, HCMC"),
    ("Quận 3", "Quan 3, HCMC"),
    ("Quận 4", "Quan 4, HCMC"),
    ("Quận 5", "Quan 5, HCMC"),
    ("Quận 7", "Quan 7, HCMC"),
    ("Quận 10", "Quan 10, HCMC"),
    ("Bình Thạnh", "Binh Thanh, HCMC"),
    ("Gò Vấp", "Go Vap, HCMC"),
    ("Tân Bình", "Tan Binh, HCMC"),
    ("Quận Tân Phú", "Quan Tan Phu, HCMC"),
    ("Bình Chánh", "Binh Chanh, HCMC"),
    ("Nhà Bè", "Nha Be, HCMC"),
    ("Đà Nẵng", "Da Nang, Da Nang"),
    ("Hải Phòng", "Hai Phong, Hai Phong"),
    ("Cần Thơ", "Ninh Kieu, Can Tho"),
    ("Biên Hòa", "Bien Hoa, Dong Nai"),
    ("Vũng Tàu", "Vung Tau, Ba Ria"),
    ("Nha Trang", "Nha Trang, Khanh Hoa"),
    ("Huế", "Hue, Thua Thien"),
    ("Quy Nhơn", "Quy Nhon, Binh Dinh"),
    ("Vinh", "Vinh, Nghe An"),
    ("Hà Nam", "Ha Nam, Ha Nam"),
]

print("\n" + "="*80)
print("BẢNG DỰ ĐOÁN GIÁ BĐS TRÊN KHẮP NƯỚC (100m², 3 phòng, 2 toilet, 2 tầng)")
print("="*80 + "\n")

results = []
api_url = "http://localhost:8000/api/predict"

for idx, (name, location) in enumerate(districts, 1):
    try:
        payload = {
            "area": 100,
            "location": location,
            "bedrooms": 3,
            "bathrooms": 2,
            "floors": 2,
            "frontage": 1,
            "legalStatus": "Sale contract",
            "furnitureState": "Fully furnished"
        }
        
        response = requests.post(api_url, json=payload)
        data = response.json()
        
        price = int(data['estimatedValue'])
        acc = data['confidence'] * 100
        
        results.append({
            'STT': idx,
            'Quận/Thành phố': name,
            'Giá dự đoán (tr VND)': price,
            'Accuracy (%)': f"{acc:.1f}%"
        })
        
        print(f"{idx:2d}. {name:<25} | {price:>8,} tr VND | {acc:.1f}%")
        
    except Exception as e:
        print(f"{idx:2d}. {name:<25} | ERROR: {str(e)}")

# Statistics
print("\n" + "="*80)
print("THỐNG KÊ\n")

prices = [r['Giá dự đoán (tr VND)'] for r in results]
avg_price = sum(prices) / len(prices)
max_price = max(prices)
min_price = min(prices)

# Find which district has max/min
max_dist = [r for r in results if r['Giá dự đoán (tr VND)'] == max_price][0]
min_dist = [r for r in results if r['Giá dự đoán (tr VND)'] == min_price][0]

print(f"📊 Giá cao nhất:    {max_price:,} tr VND ({max_dist['Quận/Thành phố']})")
print(f"📊 Giá thấp nhất:   {min_price:,} tr VND ({min_dist['Quận/Thành phố']})")
print(f"📊 Giá trung bình:  {int(avg_price):,} tr VND")
print(f"📊 Chênh lệch:      {max_price - min_price:,} triệu VND")
print(f"📊 Model Accuracy:  62.88% ± 4,826 tr VND (MAE)")

print("\n" + "="*80)
print("CÔNG THỨC TÍNH TOÁN\n")
print("""
Input (từ form):
  - Area (m²): 100
  - Location: "Quan 1, HCMC"
  - Bedrooms: 3
  - Bathrooms: 2
  - Floors: 2
  - Frontage: 1 (Yes)

Xử lý:
  1. Extract district từ location → "Quận 1"
  2. Normalize: "Quan 1" → "Quận 1"
  3. One-hot encode district → [0,0,...,1,...,0] (235 features)
  
Features vector (240 features):
  [100, 2, 3, 2, 1, 0, 0, ..., 1 (Quận 1), ..., 0]
   ^ Area, Floors, Bedrooms, Bathrooms, Frontage, ..., District features

RandomForest predict:
  - 100 decision trees vote
  - Mỗi tree predict giá dựa trên features
  - Lấy trung bình → 48,194 tr VND
  
Output:
  - Estimated Value: 48,194 tr VND
  - Confidence: 62.88% (R² score)
  - Error range: ±4,826 tr VND (MAE)

Hoạt động của từng feature:
  • Area (40.1%): Tăng 100m² → +~4-5K tr VND/m²
  • Floors (19.2%): Tăng 1 tầng → +~2-3K tr VND
  • Bedrooms (7.2%): Tăng 1 phòng → +~1-2K tr VND
  • District (5-15%): Quận 1 vs Quận 7 = 39K tr VND chênh
""")

print("="*80)
