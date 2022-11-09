from django.shortcuts import render

from model.predict import VideoDescription
from .models import Video
from rest_framework import status
from .serializers import VideoSerializer
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
        obj = video_serializer.save()
        name = obj.video.name
        vd = VideoDescription(name.split('/')[-1], 'media/' + '/'.join(name.split('/')[0:-1]))
        obj.caption = vd.extract_features_and_predict()[0]
        obj.save()
        return Response(obj.caption, status=status.HTTP_201_CREATED)
    else:
        return Response(video_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
