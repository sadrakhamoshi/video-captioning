from django.shortcuts import render
from .models import Video
from rest_framework import status
from .serializers import VideoSerializer  
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

# Create your views here.
def index(request):
    video = Video.objects.all()
    return render(request, "index.html", {"video":video})


class VideoView(APIView):

  def post(self, request, *args, **kwargs):
    video_serializer = VideoSerializer(data=request.data)
    if video_serializer.is_valid():
        video_serializer.save()
        print(type(video_serializer.data))
        return Response(video_serializer.data, status=status.HTTP_201_CREATED)
    else:
        return Response(video_serializer.errors, status=status.HTTP_400_BAD_REQUEST)