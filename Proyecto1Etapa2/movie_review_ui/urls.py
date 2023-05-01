from django.urls import path          
from .views import *   

urlpatterns = [
    path('', home, name='home'),
    path('predict', predict, name='predict'),
    path('showPrediction/<int:prediction>/<int:probability>', showPrediction, name='showPrediction'),
    path('error/<str:error>', error, name='error')
]