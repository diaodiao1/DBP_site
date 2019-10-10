from django.db import models

class Predict(models.Model):
    #预测的蛋白质序列
    text = models.TextField()