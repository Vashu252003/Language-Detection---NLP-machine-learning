# 🌍 Language Detection API

A powerful machine learning-powered REST API that automatically detects the language of text input. Built with FastAPI, scikit-learn, and deployed on Render.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

## 🚀 Live Demo

**API Base URL**: `https://language-detection-nlp-machine-learning.onrender.com`

- 📚 **Interactive Documentation**: [API Docs](https://language-detection-nlp-machine-learning.onrender.com/docs)
- ❤️ **Health Check**: [Service Status](https://language-detection-nlp-machine-learning.onrender.com/health)
- 🧪 **Quick Test**:
  ```bash
  curl -X POST "https://language-detection-nlp-machine-learning.onrender.com/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, how are you today?"}'
  ```

## ✨ Features

- 🎯 **High Accuracy**: Detects 17+ languages with confidence scores
- ⚡ **Fast Response**: < 1 second response time
- 📦 **Batch Processing**: Handle multiple texts in one request
- 🔍 **Health Monitoring**: Built-in health checks and status endpoints
- 📖 **Auto Documentation**: Interactive Swagger UI documentation
- 🌐 **Production Ready**: Deployed on Render with proper error handling
- 🐍 **Easy Integration**: Simple REST API with Python client library

## 🌍 Supported Languages

Arabic, Danish, Dutch, English, French, German, Greek, Hindi, Italian, Kannada, Malayalam, Portugeese, Russian, Spanish, Sweedish, Tamil, Turkish

_Get the complete list via API: `GET /supported-languages`_

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.12+
- **ML Framework**: scikit-learn, pandas, numpy
- **Text Processing**: TF-IDF Vectorizer with character n-grams
- **Models**: Multinomial Naive Bayes, Logistic Regression, SVM, Random Forest
- **Deployment**: Render (with automatic deployments)
- **Documentation**: Swagger UI, ReDoc

## 📊 API Endpoints

### Core Endpoints

| Method | Endpoint               | Description                     |
| ------ | ---------------------- | ------------------------------- |
| `GET`  | `/`                    | API information and status      |
| `GET`  | `/health`              | Health check and model status   |
| `POST` | `/predict`             | Single text language prediction |
| `POST` | `/batch-predict`       | Batch text language prediction  |
| `GET`  | `/supported-languages` | List all supported languages    |
| `GET`  | `/docs`                | Interactive API documentation   |

### Example Usage

#### Single Prediction

```bash
curl -X POST "https://language-detection-nlp-machine-learning
.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, comment allez-vous?"}'
```

**Response:**

```json
{
  "text": "Bonjour, comment allez-vous?",
  "predicted_language": "French",
  "confidence": 0.9876
}
```

#### Batch Prediction

```bash
curl -X POST "https://language-detection-nlp-machine-learning.onrender.com/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Bonjour monde", "Hola mundo"]}'
```

**Response:**

```json
{
  "predictions": [
    {
      "text": "Hello world",
      "predicted_language": "English",
      "confidence": 0.95
    },
    {
      "text": "Bonjour monde",
      "predicted_language": "French",
      "confidence": 0.92
    },
    {
      "text": "Hola mundo",
      "predicted_language": "Spanish",
      "confidence": 0.94
    }
  ]
}
```

## 🏗️ Project Structure

```
language-detection-api/
├── 📄 app.py                    # ML training and CLI interface
├── 🚀 api_server.py             # FastAPI server application
├── 🐍 client_example.py         # Python client examples
├── 📋 requirements.txt          # Python dependencies
├── 📊 data/
│   └── Language Detection.csv   # Training dataset
├── 🤖 saved_models/
│   └── best_langid_model.joblib # Trained ML model
└── 📖 README.md                 # Project documentation
```

## 🚦 Getting Started

### Prerequisites

- Python 3.12+
- pip package manager
- Git (for cloning)

### Local Development

1. **Clone the repository**

   ```bash
   git clone https://github.com/Vashu252003/Language-Detection-NLP-machine-learning.git
   cd Language-Detection-NLP-machine-learning
   ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**

   ```bash
   python app.py train
   ```

   This will:

   - Load training data from `kaggle`
   - Train multiple ML models with cross-validation
   - Select the best performing model
   - Save it to `saved_models/best_langid_model.joblib`

5. **Start the API server**

   ```bash
   python api_server.py
   ```

   The API will be available at `http://localhost:8000`

6. **Test the API**
   - Visit `http://localhost:8000/docs` for interactive documentation
   - Or run: `python client_example.py`

### CLI Usage (Alternative)

For command-line usage without the API:

```bash
# Train model
python app.py train

# Interactive predictions
python app.py predict
```

## 🐍 Python Client Library

Use the included client library for easy integration:

```python
from client_example import LanguageDetectionClient

# Initialize client (use your deployed URL)
client = LanguageDetectionClient("https://language-detection-nlp-machine-learning.onrender.com")

# Check API health
health = client.health_check()
print(f"API Status: {health['status']}")

# Single prediction
result = client.predict_language("Hello, world!")
print(f"Language: {result['predicted_language']}")

# Batch prediction
batch_result = client.batch_predict([
    "Hello world",
    "Bonjour monde",
    "Hola mundo"
])
for pred in batch_result['predictions']:
    print(f"'{pred['text']}' → {pred['predicted_language']}")
```

## 🌐 JavaScript/Web Integration

```javascript
// Detect language from web applications
async function detectLanguage(text) {
  const response = await fetch(
    "https://language-detection-nlp-machine-learning.onrender.com/predict",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    }
  );

  const result = await response.json();
  return result.predicted_language;
}

// Usage
detectLanguage("Hello world").then((language) => {
  console.log(`Detected language: ${language}`);
});
```

## 📈 Performance

- **Response Time**: < 1 second for single predictions
- **Batch Processing**: Up to 100 texts per request
- **Accuracy**: ~95%+ on test dataset
- **Supported Text Length**: 1-10,000 characters per text
- **Concurrent Requests**: Handles multiple simultaneous requests

## 🔧 Configuration

### Environment Variables

| Variable     | Default                                 | Description           |
| ------------ | --------------------------------------- | --------------------- |
| `PORT`       | `8000`                                  | Server port           |
| `HOST`       | `0.0.0.0`                               | Server host           |
| `MODEL_PATH` | `saved_models/best_langid_model.joblib` | Path to trained model |

### Model Configuration

The system automatically selects the best performing model from:

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest
- SGD Classifier

Using TF-IDF character n-grams (1-4) with up to 20,000 features.

## 🚀 Deployment

### Deploy to Render (Recommended)

1. **Fork this repository**
2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Create new Web Service
   - Connect your GitHub repository
3. **Configure service**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
4. **Deploy**: Render will automatically build and deploy

### Deploy with Docker

```bash
# Build image
docker build -t language-detection-api .

# Run container
docker run -p 8000:8000 language-detection-api
```

### Deploy to other platforms

The app is compatible with:

- Heroku
- Railway
- DigitalOcean App Platform
- AWS Elastic Beanstalk
- Google Cloud Run

## 🧪 Testing

Run the test suite:

```bash
# Test model directly
python model_test.py

# Test API endpoints
python client_example.py

# Manual API testing
curl http://localhost:8000/health
```

## 📊 Dataset

The model is trained on a multilingual dataset containing text samples in 17 languages. The dataset includes:

- **Size**: 10,000+ text samples
- **Languages**: 17 different languages
- **Text Types**: Various domains (news, social media, literature)
- **Preprocessing**: Normalized text with character-level features

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Format code
black .
isort .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- **scikit-learn** community for excellent ML tools
- **FastAPI** for the amazing web framework
- **Render** for reliable and free hosting
- Open source dataset contributors

## 🐛 Bug Reports & Feature Requests

Please use the [GitHub Issues](https://github.com/yourusername/language-detection-api/issues) page to report bugs or request features.

## 📚 Documentation

- **API Documentation**: Available at `/docs` endpoint
- **Model Details**: See `app.py` for training implementation
- **Deployment Guide**: Check deployment section above

## 🔄 Changelog

### v1.0.0 (2024-01-XX)

- Initial release
- Support for 17 languages
- FastAPI REST API
- Render deployment
- Batch processing support
- Interactive documentation

---

⭐ **Star this repo** if you find it useful!

🚀 **[Try the live API](https://your-app-name.onrender.com/docs)** | 📖 **[Read the docs](https://your-app-name.onrender.com/docs)** | 🐛 **[Report issues](https://github.com/yourusername/language-detection-api/issues)**
