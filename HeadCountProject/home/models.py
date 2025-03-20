from django.db import models

# Create your models here.

# Create your models here.

class UserDetails(models.Model):
    username = models.CharField(max_length=122)
    email = models.CharField(max_length=122)
    password = models.CharField(max_length=122)
    phone = models.CharField(max_length=13)