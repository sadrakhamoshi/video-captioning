import imp
import os
from uuid import uuid4
from django.db import models


def wrapper(instance, filename):
    ext = filename.split('.')[-1]
    # get filename
    filename = '{}.{}'.format(uuid4().hex, ext)
    # return the whole path to the file
    return os.path.join('video/', filename)

# Create your models here.
class Video(models.Model):
    caption = models.CharField(max_length=500)
    video = models.FileField(upload_to="video/%y")
    def __str__(self) -> str:
        return self.caption