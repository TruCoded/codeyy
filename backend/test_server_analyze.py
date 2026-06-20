import urllib.request
import json

payload = {
    "code": "def hello():\n    print('hello world')",
    "language": "python"
}

headers = {
    "Content-Type": "application/json"
}

req = urllib.request.Request(
    "http://127.0.0.1:8000/analyze",
    data=json.dumps(payload).encode('utf-8'),
    headers=headers,
    method="POST"
)

try:
    print("Sending request to /analyze...")
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode('utf-8'))
        print("Success! Response summary:")
        print(f"Keys returned: {list(res_data.keys())}")
        print(f"Time Complexity: {res_data.get('time_complexity')}")
        print(f"Space Complexity: {res_data.get('space_complexity')}")
except Exception as e:
    print(f"Failed: {e}")
