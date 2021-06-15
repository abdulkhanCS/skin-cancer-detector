from django.shortcuts import render

# Create your views here.

#View is being requested
def index(request):
    return render(request,'detector_app/index.html')

def result(request, img):
    return render(request,'detector_app/result.html')