from django.urls import path
from . import views
from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
    path('', views.index),
    path('generate', csrf_exempt(views.VideoView.as_view()), name='video-upload'),
]
