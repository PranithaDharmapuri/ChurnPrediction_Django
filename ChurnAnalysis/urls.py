from django.urls import path
from ChurnAnalysis import views


urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_dataset, name="upload_dataset"),
]