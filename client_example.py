import requests

BASE_URL = "http://localhost:8000"

def test_health():
    print("🔍 Checking health...")
    r = requests.get(f"{BASE_URL}/health")
    print("Health:", r.json())

def test_single_prediction():
    print("\n📝 Testing single prediction...")
    data = {"text": "Bonjour, comment allez-vous?"}
    r = requests.post(f"{BASE_URL}/predict", json=data)
    print("Response:", r.json())

def test_batch_prediction():
    print("\n📦 Testing batch prediction...")
    data = {"texts": [
        "Hello, how are you?",
        "Hola, ¿cómo estás?",
        "नमस्ते, आप कैसे हैं?"
    ]}
    r = requests.post(f"{BASE_URL}/batch-predict", json=data)
    print("Response:", r.json())

def test_supported_languages():
    print("\n🌍 Supported languages...")
    r = requests.get(f"{BASE_URL}/supported-languages")
    print("Response:", r.json())

if __name__ == "__main__":
    test_health()
    test_single_prediction()
    test_batch_prediction()
    test_supported_languages()
