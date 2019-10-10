from django.urls import path
from . import views

urlpatterns = [
    path('Results', views.update_predict, name='update_predict'),
]
