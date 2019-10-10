from django.urls import path
from . import views

# start with MsDBP
urlpatterns = [
    #path('', views.MsDBP, name='MsDBP'),
    # http://localhost:8000/MsDBP/readme
    path('readme', views.readme, name='readme'),
    path('download', views.download, name='download'),
    path('sample_detail', views.sample_detail, name='sample_detail'),
    
]