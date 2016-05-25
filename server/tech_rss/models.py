from django.contrib.postgres import fields
from django.db import models


class User(models.Model):
    id = models.PositiveIntegerField(primary_key=True)
    username = models.CharField(max_length=30)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    categories = fields.ArrayField(models.IntegerField())


class Site(models.Model):
    domain = models.CharField(max_length=40, primary_key=True)
    users = models.ManyToManyField(User)
