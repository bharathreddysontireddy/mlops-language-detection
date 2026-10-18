# Language Detection ML System

A Flask-based Machine Learning application that detects languages using a TF-IDF Vectorizer and Multinomial Naive Bayes classifier.

## Setup Instructions

1. Create a virtual environment and activate it:
   `python -m venv venv`
   `source venv/bin/activate` (Mac/Linux) or `.\venv\Scripts\activate` (Windows)

2. Install dependencies:
   `pip install -r requirements.txt`

3. Train the model (This generates `model.pkl`):
   `python train_model.py`

4. Run the Flask application:
   `python app.py`

5. Open `http://127.0.0.1:5000` in your browser.

## Docker Setup
Build the image: `docker build -t language-detector:1.0 .`
Run container: `docker run -p 5000:5000 language-detector:1.0`