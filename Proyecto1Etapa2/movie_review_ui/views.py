from django.shortcuts import render, redirect

import json
import joblib
import contractions
import pandas
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer

vectorizer = joblib.load('model/vectorizer.joblib')
model = joblib.load('model/modelo_random_forest.joblib')

def process_text(text):

    text.replace('[^a-zA-Z ]', '')
    text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = LancasterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens
    text = ' '.join(tokens)

    return text

def home(request):
    return render(request, 'home.html')

def predict(request):

    if request.method == 'POST':
        probabilidad = 0
        prediccion = 0

        # Obtener los datos del formulario
        text = request.POST['mensaje_para_prediccion']
        text = process_text(text)

        try:

            # Vectorizar el texto
            text_vectorized = vectorizer.transform([text])

            prediccion = model.predict(text_vectorized)[0]
            probabilidad = model.predict_proba(text_vectorized)[0][prediccion]
            probabilidad = int(probabilidad * 100)

            return redirect('showPrediction', prediccion, probabilidad)
        except Exception as e:
            return redirect('error', e)

    return render(request, 'predict.html')

def error(request, error):
    return render(request, 'error.html', {'error': error})

def showPrediction(request, prediction, probability):

    with open('model\model_metrics.json', 'r') as f:
        model_metrics = json.load(f)

    accuracy = float(model_metrics['accuracy']) * 100
    precision = float(model_metrics['precision']) * 100
    recall = float(model_metrics['recall']) * 100
    f1_score = float(model_metrics['f1']) * 100

    if prediction == 0:
        prediction = 'Clasificacion negativa'
    else:
        prediction = 'Clasificacion positiva'

    context = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'prediction': prediction,
        'probability': probability
    }
    return render(request, 'show_prediction.html', context)
