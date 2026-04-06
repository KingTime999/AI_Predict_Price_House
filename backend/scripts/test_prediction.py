import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Test data
test_payload = {
    "area": 100,
    "location": "Quan 7, TP.HCM",
    "bedrooms": 3,
    "bathrooms": 2,
    "frontage": 1,
    "floors": 2,
    "legalStatus": "Sale contract",
    "furnitureState": "Fully furnished"
}

print("=" * 70)
print("Testing Prediction API")
print("=" * 70)
print(f"\nPayload:")
print(json.dumps(test_payload, indent=2, ensure_ascii=False))

try:
    response = requests.post(
        "http://127.0.0.1:8000/api/predict",
        json=test_payload,
        timeout=10
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction Success!")
        print(f"\nResult:")
        print(f"  Estimated Value: {result['estimatedValue']:,.2f} million VND")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Trend: {result['trend']}")
        print(f"  Analysis: {result['analysis']}")
    else:
        print(f"\n✗ Error!")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n✗ Connection Error: {e}")
    print(f"Make sure backend is running with: npm run api:dev")
