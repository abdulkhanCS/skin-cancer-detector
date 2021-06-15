from django.db import models

# Create your models here.
class UserInfo(models.Model):
    detection_result = models.CharField(max_length=40)
