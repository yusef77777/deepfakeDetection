from django.db import models
from django.utils import timezone


class Email(models.Model):
    email = models.EmailField(unique=True)  # Store the email, ensuring it's unique
    created_at = models.DateTimeField(auto_now_add=True)  # Store when it was added

    def __str__(self):
        return self.email



class Feedback(models.Model):
    RATING_CHOICES = [
        (1, '1 Star'),
        (2, '2 Stars'),
        (3, '3 Stars'),
        (4, '4 Stars'),
        (5, '5 Stars'),
    ]
    
    rating = models.PositiveSmallIntegerField(choices=RATING_CHOICES)
    comment = models.TextField(blank=True, null=True)
    page_url = models.URLField()
    created_at = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    
    def __str__(self):
        return f"Feedback ({self.rating} stars)"
    

    