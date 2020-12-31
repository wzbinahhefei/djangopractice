from django.urls import path, re_path
from . import views

app_name = "collector"

urlpatterns = [
    path('', views.tank4c9, name='collector'),
    path('gettank4c9data/', views.gettank4c9data, name='gettank4c9data'),
    path('getcollectordata/', views.getcollectordata, name='getcollectordata'),
    ]

