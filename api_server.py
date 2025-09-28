from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import re
import string
from typing import Dict, List,Optional
import uvicorn

# --- Config ---
MODEL_PATH = "saved_models/best_langid_model.joblib"

# --- FastAPI App ---
app = FastAPI(
    title="Language Detection API",
    description="API for detecting the language of input text using machine learning",
    version="1.0.0"
)

# Global model variable
model = None

# --- Pydantic Models ---
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predicted_language: str
    confidence: Optional[float] = None

class BatchTextInput(BaseModel):
    texts: List[str]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str

# --- Text preprocessing (same as in your original code) ---
def normalize_text(text: str) -> str:
    """
    Normalize input text for language detection:
    - Lowercase
    - Remove emojis and rare symbols
    - Keep letters, numbers, and common punctuation
    """
    # Convert to lowercase
    text = text.lower()
    
    # Keep letters (all alphabets), numbers, punctuation, and whitespace
    text = re.sub(r"[^\w\s" + re.escape(string.punctuation) + "]", " ", text, flags=re.UNICODE)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# --- Model Loading ---
def load_model():
    """Load the trained model on startup"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            print(f"❌ Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Load model when the API starts"""
    load_model()

# --- API Endpoints ---
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Language Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global model
    if model is not None:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            message="API is running and model is loaded"
        )
    else:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            message="Model is not loaded. Please train the model first."
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_language(input_data: TextInput):
    """Predict the language of input text"""
    global model
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first or check server logs."
        )
    
    if not input_data.text.strip():
        raise HTTPException(
            status_code=400, 
            detail="Input text cannot be empty"
        )
    
    try:
        # Normalize the input text
        clean_text = normalize_text(input_data.text)
        
        # Make prediction
        prediction = model.predict([clean_text])
        
        # Get prediction probabilities if available
        confidence = None
        if hasattr(model.named_steps['clf'], 'predict_proba'):
            try:
                probabilities = model.predict_proba([clean_text])
                confidence = float(max(probabilities[0]))
            except:
                pass  # Some models might not support predict_proba
        
        return PredictionResponse(
            text=input_data.text,
            predicted_language=prediction[0],
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )
    

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict_language(input_data: BatchTextInput):
    """Predict languages for multiple texts"""
    global model
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first or check server logs."
        )
    
    if not input_data.texts:
        raise HTTPException(
            status_code=400, 
            detail="Input texts list cannot be empty"
        )
    
    if len(input_data.texts) > 100:  # Limit batch size
        raise HTTPException(
            status_code=400, 
            detail="Batch size cannot exceed 100 texts"
        )
    
    try:
        predictions = []
        
        # Process each text
        clean_texts = [normalize_text(text) for text in input_data.texts]
        batch_predictions = model.predict(clean_texts)
        
        # Get probabilities if available
        confidences = []
        if hasattr(model.named_steps['clf'], 'predict_proba'):
            try:
                probabilities = model.predict_proba(clean_texts)
                confidences = [float(max(prob)) for prob in probabilities]
            except:
                confidences = [None] * len(input_data.texts)
        else:
            confidences = [None] * len(input_data.texts)
        
        # Create response
        for i, text in enumerate(input_data.texts):
            predictions.append(PredictionResponse(
                text=text,
                predicted_language=batch_predictions[i],
                confidence=confidences[i]
            ))
        
        return BatchPredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/supported-languages")
async def get_supported_languages():
    """Get list of supported languages (requires trained model)"""
    global model
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Get unique classes from the model
        if hasattr(model.named_steps['clf'], 'classes_'):
            languages = model.named_steps['clf'].classes_.tolist()
        else:
            raise HTTPException(
                status_code=500, 
                detail="Cannot retrieve supported languages from this model"
            )
        
        return {
            "supported_languages": sorted(languages),
            "total_count": len(languages)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving supported languages: {str(e)}"
        )

# --- Error Handlers ---
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "Please check the API documentation at /docs"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Something went wrong on the server"}

# --- Run Server ---
if __name__ == "__main__":
    print("🚀 Starting Language Detection API server...")
    print("📚 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Alternative docs at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "api_server:app",  # Change this to match your filename
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes during development
        log_level="info"
    )