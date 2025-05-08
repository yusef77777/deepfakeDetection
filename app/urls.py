from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import submit_feedback
from . import views

urlpatterns = [  # ← now this handles /
    path('', views.upload_video, name='upload_video'),
    path('cancel_processing/', views.cancel_processing, name='cancel_processing'),
    path('api/feedback/', submit_feedback, name='submit_feedback'),
    path('newsletter/submit-email/', views.email_submission, name='submit_email'),

 

]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


