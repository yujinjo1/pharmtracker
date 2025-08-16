# models.py
import openai
import os
from django.db import models

class DrugModel(models.Model):
   name = models.CharField(max_length=255)
   image_url = models.URLField()
   ingredients = models.TextField()
   indications = models.TextField()
   dosage = models.TextField()
   precautions = models.TextField()

   def __str__(self):
       return self.name