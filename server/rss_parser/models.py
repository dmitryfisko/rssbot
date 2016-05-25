from django.db import models


# Create your models here.

class Post(models.Model):
    category = models.IntegerField()
    text_hash = models.CharField(max_length=30)
