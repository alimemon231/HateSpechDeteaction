from django.urls import path
from .views import HateSpeachFinder

urlpatterns = [

    path("" ,  HateSpeachFinder.as_view()),
]