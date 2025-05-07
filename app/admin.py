

from django.contrib import admin
from .models import Feedback
from .models import Email

@admin.register(Email)
class EmailAdmin(admin.ModelAdmin):
    list_display = ('email', 'created_at')  # Display email and date of creation
    search_fields = ('email',)  # Make the email field searchable



@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('rating', 'created_at', 'page_url', 'ip_address')
    list_filter = ('rating', 'created_at')
    search_fields = ('comment', 'page_url')
    readonly_fields = ('created_at', 'ip_address')