from django.shortcuts import render
from predict.models import Predict

def MsDBP(request):
    context = {}
    return render(request, 'MsDBP.html', context)

def readme(request):
    context = {}
    return render(request, 'readme.html', context)

def download(request):
    context = {}
    return render(request, 'download.html', context)

def sample_detail(request):
    context = {}
    return render(request, 'sample_detail.html', context)






