import requests

# ----------------------------
# Change this to your Render app's public URL
BASE_URL = "https://language-detection-nlp-machine-learning.onrender.com"
# ----------------------------

def health_check():
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print("Health:", resp.json())
    except Exception as e:
        print("Health check error:", e)

def predict(text: str):
    try:
        resp = requests.post(f"{BASE_URL}/predict", json={"text": text})
        print(f"Text: '{text}' → Prediction:", resp.json())
    except Exception as e:
        print("Prediction error:", e)

def batch_predict(texts):
    try:
        resp = requests.post(f"{BASE_URL}/batch-predict", json={"texts": texts})
        print("Batch prediction result:", resp.json())
    except Exception as e:
        print("Batch prediction error:", e)

def supported_languages():
    try:
        resp = requests.get(f"{BASE_URL}/supported-languages")
        print("Supported languages:", resp.json())
    except Exception as e:
        print("Error fetching supported languages:", e)

if __name__ == "__main__":
    print("🔍 Checking health...")
    health_check()
    
    print("\n📝 Testing single prediction...")
    predict("Hello, how are you today?")
    predict("Bonjour, comment allez-vous?")
    predict("Hola, ¿cómo estás?")
    
    print("\n📦 Testing batch prediction...")
    batch_predict([
        "This is English text",
        "Ceci est du texte français",
        "Dies ist deutscher Text"
    ])
    
    print("\n🌍 Supported languages...")
    supported_languages()
