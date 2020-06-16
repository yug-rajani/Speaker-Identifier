from django.forms import ModelForm

from speaker.models import FileModel


class FileForm(ModelForm):
    class Meta:
        model = FileModel
        fields = ['file']
