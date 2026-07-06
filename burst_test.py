# burst_test.py
import requests
import time

URL = "http://localhost:8000/predict"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "dev-secret-key-123" # Authenticate past Touchdown 1
}
PAYLOAD = {
    "Amount": 15.5, **{f"V{i}": 0.0 for i in range(1, 29)}
}

print("🚀 Simulating rapid incoming traffic burst...")
for i in range(35): # Send 35 rapid requests to breach the 30/min cap
    response = requests.post(URL, json=PAYLOAD, headers=HEADERS)
    print(f"Request {i+1:02d} | Status: {response.status_code} | Response: {response.text}")
    if response.status_code == 429:
        print("\n🛑 Success! Rate limiting caught the traffic burst.")
        break