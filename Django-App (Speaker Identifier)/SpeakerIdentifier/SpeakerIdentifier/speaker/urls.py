from django.urls import path
from django.conf.urls import url


from speaker.views import Predict
from speaker.views import FileView
from speaker.views import FileDeleteView

app_name = 'speaker.'

urlpatterns = [
    url(r'^predict/$', Predict.as_view(), name='APIpredict'),
    url(r'^upload/$', FileView.as_view(), name='APIupload'),
    url(r'^delete/$', FileDeleteView.as_view(), name='APIdelete'),
]
