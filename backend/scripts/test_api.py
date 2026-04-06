import requests
import json

print("Testing API endpoint...")

try:
    response = requests.get("http://localhost:8000/api/properties?limit=2")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
