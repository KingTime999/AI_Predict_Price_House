import requests
import json

try:
    resp = requests.get('http://localhost:8000/api/properties?limit=2', timeout=5)
    print(f'Status Code: {resp.status_code}')
    data = resp.json()
    print(f'Total properties: {data.get("total")}')
    print(f'Items returned: {len(data.get("items", []))}')
    if data.get('items'):
        item = data['items'][0]
        print(f'\nFirst item:')
        print(f'  ID: {item.get("id")}')
        print(f'  Title: {item.get("title")}')
        print(f'  Location: {item.get("location")}')
        print(f'  Price: {item.get("price")} million VND')
        print(f'  Image: {item.get("image")}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
