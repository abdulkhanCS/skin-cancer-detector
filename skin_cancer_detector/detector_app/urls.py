from django.contrib import admin
from django.urls import path
from . import views

#What views to load based on the url
urlpatterns = [

path('',views.index,name='index'),
path('results/',views.results,name='results'),

]