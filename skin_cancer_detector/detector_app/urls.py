from django.contrib import admin
from django.urls import path
from . import views

#What views to load based on the url
urlpatterns = [

path('',views.index,name='index'),
path('result/<img>',views.result,name='result')

]