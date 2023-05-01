from django.urls import path          
from .views import *   

urlpatterns = [
    path('', ModelMetrics.as_view(), name='api'),
    path('metrics', ModelMetrics.as_view(), name='ModelMetrics'),
    path('predict', Predict.as_view(), name='Predict'),
    path('train', Train.as_view(), name='Train'),
]