from django.urls import path
from . import views

app_name='stockapp'

urlpatterns=[
    path('',views.index,name='index'),
    path('analyze/',views.analyze,name='analyze'),
]