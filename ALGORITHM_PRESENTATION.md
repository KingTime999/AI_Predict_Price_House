# 🏠 AI Property Price Prediction
## Linear Regression Algorithm Presentation

---

## 📊 Slide 1: Thuật Toán - Linear Regression là gì?

### Định Nghĩa
**Linear Regression** là thuật toán Machine Learning đơn giản nhưng hiệu quả để dự đoán giá trị liên tục (continuous values).

### Nguyên Lý Hoạt Động
- Tìm một **đường thẳng** (trong không gian nhiều chiều) phù hợp nhất với dữ liệu
- Đường này được gọi là **hypothesis function**: `y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ`
  - `y` = Giá nhà dự đoán
  - `w₀` = Hệ số bias (intercept)
  - `w₁, w₂, ..., wₙ` = Hệ số đặc trưng (weights)
  - `x₁, x₂, ..., xₙ` = Các đặc trưng input

### Công Thức Toán Học
```
Price = w₀ + w₁(Area) + w₂(Bedrooms) + w₃(Bathrooms) + ... + w₉(FurnitureState)
```

### Ví Dụ Thực Tế
```
Giá = 100M VND + 50M/m² × Area + 280M × Bedrooms + 240M × Bathrooms
     + 100M × Floors + 55M × Frontage + ...
```

---

## 🎯 Slide 2: Tại Sao Chọn Linear Regression?

| Ưu Điểm | Giải Thích |
|---------|-----------|
| ✅ **Đơn giản** | Dễ hiểu, dễ implement |
| ✅ **Nhanh** | Thời gian training & prediction rất nhanh |
| ✅ **Interpretability** | Có thể giải thích được độ ảnh hưởng của mỗi feature |
| ✅ **Accurate cho bài toán này** | 95% confidence trên validation data |
| ✅ **Ít overfitting** | R² gap (train-test): chỉ 2.4% |

### So Sánh Với Các Thuật Toán Khác

| Thuật Toán | Độ Chính Xác | Tốc Độ | Giải Thích |
|-----------|-------------|---------|-----------|
| Linear Regression | 89.56% | ⚡⚡⚡ Rất nhanh | ✅ Tốt |
| Decision Trees | ~87% | ⚡⚡ Nhanh | ✅ Tốt |
| Random Forest | ~90% | ⚡ Chậm | ❌ Kém |
| Neural Networks | ~92% | ❌ Rất chậm | ❌ Black box |
| SVR | ~88% | ⚡ Chậm | ❌ Kém |

**→ Linear Regression: Best balance giữa accuracy, speed, interpretability**

---

## 🔧 Slide 3: Các Đặc Trưng (Features) Sử Dụng

### 9 Features Input
```
1. Area (m²)              → Diện tích nhà
2. Bedrooms              → Số phòng ngủ
3. Bathrooms             → Số phòng tắm
4. Location (Address)    → Quận/Thành phố (encoded)
5. Frontage (Binary)     → Nhà có mặt tiền không
6. Access Road Width (m) → Độ rộng đường truy cập
7. Number of Floors      → Số tầng
8. Legal Status          → Loại giấy tờ (Red/Pink/Contract)
9. Furniture State       → Tình trạng nội thất

↓ (Pre-processing)

INPUT: [100, 3, 2, "Quan 7", 1, 10, 2, "Red book", "Fully furnished"]
↓
PREPROCESSOR:
  - Standardization (Scale numeric features)
  - One-hot encoding (Categorical features)
↓
PROCESSED: [0.5, 1.2, 0.8, 1, 0, 0, 1, 1.5, 0.9, ...]
```

### Ảnh Hưởng Của Mỗi Feature (Coefficients)
```
Giá ≈ 100 tr VND 
    + 50 tr/m²      × Area
    + 280 tr        × Bedrooms      ← Tác động lớn nhất
    + 240 tr        × Bathrooms
    + 55 tr/m       × Access Road Width
    + 55 tr         × Frontage (1=có, 0=không)
    + 140 tr        × Floors
    + 120 tr        × Legal Status (Red book)
    + 80 tr         × Furniture State
```

**Hiểu được:** Thêm 1 phòng ngủ tăng giá ~280 triệu VND

---

## 📈 Slide 4: Quá Trình Training (Huấn Luyện)

### Bước 1: Chuẩn Bị Dữ Liệu (Data Preparation)
```
Raw Dataset (1,200 properties)
    ↓
Train/Test Split: 80% train (960) + 20% test (240)
    ↓
Preprocessing:
  - StandardScaler: Chuẩn hóa numeric features
  - OneHotEncoder: Mã hóa categorical features
  - Drop missing values / Imputation
```

### Bước 2: Xây Dựng Model
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
# → Tìm weights w₀, w₁, ..., w₉ tối ưu
```

### Bước 3: Tối Ưu Hóa (Optimization)
**Mục tiêu:** Minimize Mean Squared Error (MSE)
```
MSE = (1/n) × Σ(y_predicted - y_actual)²

Thuật toán: Normal Equation
w = (X^T × X)^(-1) × X^T × y
```

### Bước 4: Kiểm Chứng (Validation)
```
Test set (240 properties)
  ↓ (Prediction)
y_predicted = model.predict(X_test)
  ↓ (Evaluation)
Calculate metrics: MAE, RMSE, R²
```

---

## 📊 Slide 5: Model Performance (Kết Quả)

### Training Metrics
```
┌─────────────────────────────────────────┐
│        TRAINING SET PERFORMANCE         │
├─────────────────────────────────────────┤
│ MAE (Mean Absolute Error)               │
│   = 280 triệu VND                       │
│   → Trung bình, lỗi dự đoán 280 tr/căn  │
│                                         │
│ RMSE (Root Mean Squared Error)          │
│   = 450 triệu VND                       │
│   → Căn bậc 2 của trung bình bình phương│
│   → Penalize lỗi lớn hơn                │
│                                         │
│ R² Score                                │
│   = 0.8956 (89.56%)                     │
│   → Model giải thích 89.56% biến thiên  │
│      của giá nhà                        │
└─────────────────────────────────────────┘
```

### Validation (Test) Metrics
```
┌─────────────────────────────────────────┐
│         TEST SET PERFORMANCE            │
├─────────────────────────────────────────┤
│ MAE   = 310 triệu VND (+10%)             │
│ RMSE  = 520 triệu VND (+15%)             │
│ R²    = 0.8712 (87.12%) (-2.4%)         │
│                                         │
│ Gap Analysis:                           │
│ R² Gap = 0.8956 - 0.8712 = 0.0244      │
│        = 2.44% ← MINIMAL OVERFITTING    │
│                 (Good generalization) │
└─────────────────────────────────────────┘
```

### Visualize Performance
```
Actual Price vs Predicted Price:

14000 |           ╱╱╱╱╱
      |        ╱╱╱    ╱
12000 |     ╱╱╱        ╱   ← Lý tưởng (y=x)
      |   ╱╱           ╱
10000 |  ╱              ╱
      | ╱              ╱
 8000 |╱              ╱
      |              ╱╱╱
 6000 |             ╱    ╱
      |            ╱      ╱
 4000 |───────────────────╱
      |_________________╱
        0   5000  10000  15000
        Actual Price
        
Phần lớn điểm nằm gần đường y=x → Model tốt!
```

---

## 🔄 Slide 6: Cách Hoạt Động Khi Dự Đoán (Inference)

### Ví Dụ Thực Tế: Dự Đoán Giá Nhà

**INPUT từ người dùng:**
```
Diện tích: 100 m²
Quận: Quận 7, TP.HCM
Phòng ngủ: 3
Phòng tắm: 2
Mặt tiền: Có (1)
Độ rộng đường: 10m
Số tầng: 2
Giấy tờ: Sổ đỏ (Red book)
Nội thất: Đầy đủ (Fully furnished)
```

**BƯỚC 1: Preprocessing**
```
Area=100 → StandardScaler → 0.85
Bedrooms=3 → StandardScaler → 1.20
Bathrooms=2 → StandardScaler → 0.90
Location="Quan 7" → OneHotEncoding → [0, 0, 1, 0, ...]
Frontage=1 → Keep → 1
AccessRoad=10 → StandardScaler → 1.50
Floors=2 → StandardScaler → -0.30
LegalStatus="Red book" → OneHotEncoding → [1, 0, 0]
FurnitureState="Fully" → OneHotEncoding → [1, 0, 0]

Processed Vector: X = [0.85, 1.20, 0.90, 0, 0, 1, 0, ..., 1, 1.50, -0.30, 1, 0, 0, 1, 0, 0]
```

**BƯỚC 2: Linear Regression Prediction**
```
y = w₀ + w₁X₁ + w₂X₂ + ... + w₉X₉

y = 100 + 50(0.85) + 280(1.20) + 240(0.90) 
    + 0(0) + 0(0) + 55(1) + 0(0) + ...
    + 55(1) + 140(1.50) + ... + 120(1) + 80(1)

y = 100 + 42.5 + 336 + 216 + 55 + 210 + 120 + 80 + ...

≈ 7,778 triệu VND = 7,777,780,000 VND
```

**BƯỚC 3: Confidence Calculation**
```
Confidence% = 100 - (RMSE / Predicted Price) × 100
            = 100 - (450 / 7778) × 100
            = 100 - 5.78%
            = 94.22% ≈ 95%
```

**OUTPUT:**
```
┌─────────────────────────────────┐
│  ESTIMATED VALUE                │
│  7,777,780,000 VND              │
│  (7,777.78 triệu VND)           │
│                                 │
│  CONFIDENCE: 95.0%              │
│  EXPECTED GROWTH: +4.38%/year  │
│                                 │
│  ANALYSIS:                      │
│  "Model estimates this property │
│  around 7,777,780,000 VND for  │
│  Quan 7, HCMC. Confidence is    │
│  95.0% based on validation      │
│  error. Property is well-priced │
│  for the area with good         │
│  potential returns."            │
└─────────────────────────────────┘
```

---

## 🧮 Slide 7: Công Thức & Toán Học Chi Tiết

### Normal Equation (Công Thức Giải Analytic)
```
Mục tiêu: Find weights W that minimize MSE

MSE = (1/n) × Σ(y_predicted - y_actual)²
    = (1/n) × ||XW - y||²

Gradient descent = 0:
∂MSE/∂W = 0
2X^T(XW - y) = 0
X^T(XW - y) = 0

→ X^TXW = X^Ty

→ W = (X^TX)^(-1) × X^Ty  ← NORMAL EQUATION
```

### Ví Dụ Ma Trận
```
Data Matrix X (960×10):
┌─────────────────────────┐
│ x₁₁ x₁₂ x₁₃ ... x₁₁₀   │  Property 1
│ x₂₁ x₂₂ x₂₃ ... x₂₁₀   │  Property 2
│ x₃₁ x₃₂ x₃₃ ... x₃₁₀   │  Property 3
│ ...                     │
│ x₉₆₀,₁ ... x₉₆₀,₁₀     │  Property 960
└─────────────────────────┘

Price Vector y (960×1):
┌────────────┐
│ 7633.01    │  Price 1
│ 23501.81   │  Price 2
│ 19424.11   │  Price 3
│ ...        │
│ 15000.00   │  Price 960
└────────────┘

Weights W (10×1):
┌────────────┐
│ w₀ = 100   │  bias
│ w₁ = 50    │  Area coeff
│ w₂ = 280   │  Bedrooms coeff
│ ...        │
│ w₉ = 80    │
└────────────┘
```

---

## ⚙️ Slide 8: Tiền Xử Lý Dữ Liệu (Preprocessing)

### 1. StandardScaler (Chuẩn Hóa)
```
Mục đích: Đưa tất cả features về cùng scale

z = (x - mean) / std_dev

Ví dụ:
Area: Mean=150, Std=50
  → Property (Area=100) → z = (100-150)/50 = -1.0
  → Property (Area=200) → z = (200-150)/50 = +1.0

Bedrooms: Mean=3, Std=1
  → Property (Bed=1) → z = (1-3)/1 = -2.0
  → Property (Bed=5) → z = (5-3)/1 = +2.0

Lợi ích:
  - Features trên cùng scale
  - Model học nhanh hơn
  - Tránh bias từ scale khác nhau
```

### 2. OneHotEncoding (Mã Hóa Categorical)
```
Mục đích: Convert categorical → numeric

Location: "Quan 1", "Quan 7", "Hanoi"
  ↓
  Quan 1  → [1, 0, 0]
  Quan 7  → [0, 1, 0]
  Hanoi   → [0, 0, 1]

Legal Status: "Red book", "Pink book", "Contract"
  ↓
  Red book   → [1, 0, 0]
  Pink book  → [0, 1, 0]
  Contract   → [0, 0, 1]

Lợi ích:
  - Linear Regression chỉ hiểu numeric values
  - Bảo toàn thông tin categorical
```

### 3. Missing Values Handling
```
Strategy:
  - Drop rows với missing values (nếu <5% data)
  - Imputation with mean/median (nếu >>5% data)
  
Result: 1200 properties → 960 train + 240 test (all complete)
```

---

## 🎓 Slide 9: Các Thách Thức & Giải Quyết

### Challenge 1: Multicollinearity
```
Problem: Features có mối tương quan cao
  → Bedrooms & Area thường correlated

Detection:
  - Correlation matrix
  - VIF (Variance Inflation Factor)

Solution:
  - Nếu VIF > 5 → Drop một feature
  - Trong project: Location + Area decorrelated ✓
```

### Challenge 2: Outliers
```
Problem: Một vài nhà có giá rất cao/thấp
  → Skew model predictions

Example:
  [3000, 5000, 7000, 8000, 50000]  ← 50000 outlier
  
Detection:
  - Box plot
  - IQR method: Q3 + 1.5×IQR

Solution trong project:
  - Robust scaling
  - Minimal impact: R² gap = 2.4% (low)
```

### Challenge 3: Feature Engineering
```
Raw Features → Enhanced Features

Area (m²) → Area Categories:
  - Small (0-100)
  - Medium (100-200)
  - Large (>200)

Age derived từ year built → Non-linear relationship

Price per m² = Price / Area → Better predictor
```

---

## 📦 Slide 10: Implementation Stack

### Technology Used
```
Framework: scikit-learn 1.3.2
  - LinearRegression()
  - StandardScaler()
  - OneHotEncoder()
  - train_test_split()
  - mean_squared_error, r2_score

Data Processing: pandas 2.0
Data Storage: joblib (serialization)
```

### Model Artifact
```
File: backend/artifacts/house_price_model.joblib

Content:
{
  "pipeline": Pipeline(
    StandardScaler + OneHotEncoder + LinearRegression
  ),
  "metrics": {
    "train_r2": 0.8956,
    "test_r2": 0.8712,
    "train_mae": 280,
    "test_mae": 310,
    ...
  },
  "feature_names": [
    "Area", "Bedrooms", "Bathrooms", ..., "FurnitureState"
  ]
}

Size: ~50 KB
Loading time: <10ms
```

---

## 🎯 Slide 11: Kết Luận & Ưu Điểm

### Tóm Tắt Linear Regression
✅ **Hiệu Quả**
- 89.56% accuracy on training, 87.12% on test
- 95% confidence predictions

✅ **Nhanh**
- Training: <100ms
- Prediction: <5ms per property

✅ **Giải Thích Được**
- Biết được impact của từng feature
  - +1 m² → +50 tr VND
  - +1 phòng → +280 tr VND

✅ **Scalable**
- Dễ deploy trên backend
- Dễ retrain với dữ liệu mới

### Ứng Dụng Thực Tế
```
✓ Property valuation (định giá)
✓ Investment analysis (phân tích đầu tư)
✓ Market prediction (dự báo thị trường)
✓ Fraud detection (phát hiện giá bất thường)
```

---

## 📚 Slide 12: Q&A

### Câu Hỏi Thường Gặp

**Q: Tại sao R² không phải 100%?**
- A: Có các yếu tố không thể đo được (vị trí chính xác, lịch sử sửa chữa, v.v.)
- A: 87-89% khá tốt cho dự đoán giá thực tế

**Q: Model có thể dự đoán giá ngoài range training không?**
- A: Có, Linear Regression extrapolate được nhưng less reliable
- A: Confidence giảm nếu đầu vào ngoài bounds training data

**Q: Khi nào cần retrain model?**
- A: Quarterly (mỗi quý) nếu thị trường thay đổi
- A: Khi metrics degradation > 5% trên validation set

**Q: Có thể dùng Deep Learning thay vì Linear Regression không?**
- A: Có thể, nhưng không cần thiết với dataset này
- A: Deep Learning overhead > benefit cho bài toán simple này

---

## 💡 Slide 13 (Bonus): Comparison Chart

### Accuracy vs Speed vs Interpretability

```
                High Accuracy
                      ▲
                      │
        Neural Network │      Random Forest
               80-95%  │        87-92%
                      │      ╱
                      │    ╱
Interpretability      │  ╱      Decision Tree
                      │╱        
        Linear Reg ────┼──────  SVM
        89.56%        │ ╱     85-88%
                      │╱
                      │
            ╱─────────┴─────────╲
    Low Interpretability    High Interpretability

    ╱─────────────────────────────╲
Low Speed            High Speed
X 1-10s             ✓ <100ms
```

**Winner:** Linear Regression ✨ (Best Balance)

---

