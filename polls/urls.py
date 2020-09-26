from django.urls import path
from django.conf.urls import url

from . import views
from django.views.generic.base import TemplateView


app_name='polls'
urlpatterns = [
    # ex: /polls/
    path('', TemplateView.as_view(template_name='index.html'), name='index'),
    path('index', TemplateView.as_view(template_name='index.html'), name='index'),
    path('upload', TemplateView.as_view(template_name='upload.html'), name='upload'),
    path('dental', TemplateView.as_view(template_name='dental.html'), name='dental'),
    path('login', TemplateView.as_view(template_name='login.html'), name='login'),
    path('signup', TemplateView.as_view(template_name='signup.html'), name='signup'),
    path('skin', TemplateView.as_view(template_name='skin.html'), name='skin'),
    # ex: /polls/5
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    # ex: /polls/5/results
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
    # ex: /polls/5/vote
    path('<int:question_id>/vote/', views.vote, name='vote'),

    url(r'^uploads/simple/$', views.simple_upload, name='simple_upload'),
]
