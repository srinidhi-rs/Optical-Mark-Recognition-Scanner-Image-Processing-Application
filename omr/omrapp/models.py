# omrapp/models.py

from django.db import models

class OMRSheet(models.Model):
    image = models.ImageField(upload_to='omr_sheets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"OMR Sheet {self.id} - Uploaded at {self.uploaded_at}"

class Result(models.Model):
    omr_sheet = models.ForeignKey(OMRSheet, on_delete=models.CASCADE)
    score = models.FloatField()
    processed_at = models.DateTimeField(auto_now_add=True)
    final_image = models.ImageField(upload_to='final_images/', null=True, blank=True)
    stacked_image = models.ImageField(upload_to='stacked_images/', null=True, blank=True)

    def __str__(self):
        return f"Result for OMR Sheet {self.omr_sheet.id} - Score: {self.score}"

class AnswerKey(models.Model):
    question_number = models.IntegerField()
    correct_answer = models.IntegerField()

    def __str__(self):
        return f"Question {self.question_number} - Correct Answer: {self.correct_answer}"