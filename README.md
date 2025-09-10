Language Detection - NLP Machine Learning
A simple language detection system built using Python and scikit-learn, capable of identifying the language of a given text input. It supports interactive predictions and can be trained on a custom dataset.

Table of Contents
Features

Requirements

Dataset

Installation

Usage

Train the Model

Interactive Prediction

Project Structure

License

Features
Character-level TF-IDF vectorization for robust language detection.

Multiple model candidates:

Multinomial Naive Bayes

Logistic Regression

Linear SVC

Random Forest

SGD Classifier

Cross-validation to select the best model.

Interactive CLI to predict language in real time.

Preprocessing handles:

Uppercase/lowercase normalization

Punctuation

Emojis and rare symbols

Requirements
Python 3.8+

Libraries:

pandas
scikit-learn
joblib

Install dependencies:

pip install pandas scikit-learn joblib

Dataset
The project expects a CSV file with at least two columns:

Text → the text to detect language from

Language → the corresponding language label

Example path used in the code: data/Language Detection.csv

Installation
Clone the repository:

git clone <your-repo-url>
cd <your-repo-folder>

Ensure the dataset exists in the data/ folder.

Install Python dependencies (see Requirements).

Usage
Train the Model
Train the language detection model on your dataset:

python app.py train

The script performs cross-validation on multiple models. The best model is trained on the full dataset and saved to: saved_models/best_langid_model.joblib.

Interactive Prediction
Predict the language of any text interactively:

python app.py predict

Type your text and press Enter to get the predicted language. Supports uppercase letters, punctuation, and emojis. Type quit or exit to stop the interactive session.

Example:

Enter text to predict (or type 'quit' to exit): Bonjour, ça va? 😊
'Bonjour, ça va? 😊' → Predicted Language: French

Project Structure
.
├── app.py # Main application script
├── data/
│ └── Language Detection.csv # Dataset
├── saved_models/
│ └── best_langid_model.joblib # Saved trained model
└── README.md # Project documentation
