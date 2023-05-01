from django.shortcuts import render

import os
import json
import joblib
import numpy
import pandas
import contractions
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
# Vectorización de texto
from sklearn.feature_extraction.text import TfidfVectorizer

# Métricas
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from rest_framework import views
from rest_framework import status
from rest_framework.response import Response

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

class ModelMetrics(views.APIView):
    with open('model\model_metrics.json', 'r') as f:
        model_metrics = json.load(f)

    # Return json with the model metrics
    def get(self, request):
        return Response(self.model_metrics, status=status.HTTP_200_OK)

class Predict(views.APIView):
    def post(self, request):
        predictions = []

        for entry in request.data:
            try:
                # Vectorize text
                vectorized_text = vectorizer.transform([entry['text']])

                # Predict
                prediction = model.predict(vectorized_text)[0]
                predictions.append({'text': entry['text'], 'sentimiento': prediction})

            except Exception as e:
                return Response({'error': f'Error al procesar el texto {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

        return Response(predictions, status=status.HTTP_200_OK)

class Train(views.APIView):
    def post(self, request):
        # Create dataframe 
        df = pandas.DataFrame(columns=['text', 'sentimiento'])

        try:
            for entry in request.data:
                texto = entry['text']
                texto = process_text(texto)
                clasification = entry['sentimiento']
                if clasification == 'positivo':
                    clasification = 1
                elif clasification == "negativo": 
                    clasification = 0
                else:
                     return Response({'error': f'Valor de clasificación no valido: {clasification}'}, status=status.HTTP_400_BAD_REQUEST)         
                fila = [texto, clasification]
                df.loc[len(df)] = fila
            
            X = df["text"]
            Y = df["sentimiento"]

            vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
            X_count = vectorizer.fit_transform(X)

            model.fit(X_count, Y)

            # Import test data from model/processed_data.csv
            db_route = 'model/processed_data_min.csv'
            df_originales = pandas.read_csv(db_route, encoding = 'ISO-8859-1')

            df_originales.dropna(inplace=True)

            X_test = df_originales['tokens']
            Y_test = df_originales['sentimiento']
            X_count_test = vectorizer.transform(X_test)

            predictions = model.predict(X_count_test)

            # Create a json with the previous values
            model_json = {
                'accuracy': accuracy_score(Y_test, predictions),
                'f1': f1_score(Y_test, predictions, average='weighted'),
                'precision': precision_score(Y_test, predictions, average='weighted'),
                'recall': recall_score(Y_test, predictions, average='weighted')
            }

            # Save the json in a file
            with open('model/model_metrics.json', 'w') as outfile:
                json.dump(model_json, outfile)

            # Save model with joblib
            joblib.dump(model, 'model/modelo_random_forest.joblib')
            joblib.dump(vectorizer, 'model/vectorizer.joblib')

            return Response(model_json, status=status.HTTP_200_OK)


        except Exception as e:
                return Response({'error': f'Error en el entrenamiento del modelo {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

