from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.staticfiles.storage import staticfiles_storage
from .apps import SpeakerConfig
from django.conf import settings
import io
import os
import tensorflow
from tensorflow import keras
from keras.models import load_model
import pathlib
from pathlib import Path
import speech_recognition as sr
import matplotlib.pyplot as plt
from matplotlib import pylab
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from pydub import AudioSegment
from pydub.silence import split_on_silence
import html

import librosa
import numpy as np
import urllib
import base64

from speaker.serialize import FileSerializer
from speaker.models import FileModel
from django.views.generic.edit import CreateView
from django.views.generic import TemplateView
from django.views.generic import ListView


from os import listdir
from os.path import join
from os.path import isfile
import requests

from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework.parsers import FormParser
from rest_framework.generics import get_object_or_404
from rest_framework import status
from rest_framework import views


# def index(request):
#     if request.method == 'POST':
#         form = UploadFileForm(request.POST, request.FILES)
#         if form.is_valid():
#             print("check4")
#             handle_uploaded_file(request.FILES['file'])
#             return HttpResponseRedirect('/speaker')
#     else:
#         print("check5")
#         form = UploadFileForm()
#     return render(request, 'index.html', {'form': form})


# def handle_uploaded_file(f):
#     spk = getSpeaker(f)
#     print("speker is")
#     print(spk)


# def results(request, question_id):
#     response = "You're looking at the results of question %s."
#     return HttpResponse(response % question_id)


# def getSpeaker(f):

#     test_X, sample_rate = librosa.load(f, res_type='kaiser_fast')

#     mfccs = np.mean(librosa.feature.mfcc(
#         y=test_X, sr=sample_rate, n_mfcc=40).T, axis=0)

#     test_X = np.array(mfccs)
#     test_X = test_X.reshape((1, 40))

#     model_name = 'speakerIdentifier.h5'
#     model = load_model(
#         os.path.join(settings.MODEL_ROOT, model_name))

#     predicted_label = model.predict(test_X)
#     print(predicted_label.argmax())

#     labels = [' A. E. Maroney', ' Andrew NG', ' Anya', ' Arielle Lipshaw',
#               ' Betty Chen', ' Bill Mosley', ' BookAngel7', ' Brendan Hodge',
#               ' Brian von Dedenroth', ' Cata', ' Christie Nowak',
#               ' David Mecionis', ' David Mix', ' Doug', ' E. Tavano', ' Hilara',
#               ' Jean Bascom', ' Jeana Wei', ' Jennifer Wiginton',
#               ' JenniferRutters', ' JenniferW', ' Jill Engle', ' John Rose',
#               ' JudyGibson', ' Julie VW', ' JustinJYN', ' Kathy Caver',
#               ' Lisa Meyers', ' M. Bertke', ' Malone', ' Mark Nelson',
#               ' Mark Welch', ' Mary J', ' Michael Packard', ' Moromis',
#               ' Nelly ()', ' Nicodemus', ' Peter Eastman', ' President Lethe',
#               ' Ransom', ' Renata', ' Russ Clough', ' S R Colon',
#               ' Scott Walter', ' Sharon Bautista', ' Simon Evers',
#               ' Stephen Kinford', ' Steven Collins', ' Susan Hooks', ' Tonia',
#               ' VOICEGUY', ' WangHaojie', ' Wayne Donovan', ' Wendy Belcher',
#               ' Winston Tharp', ' aquielisunari', ' ashleyspence', ' badey',
#               ' calystra', ' camelot2302', ' chocmuse', ' dexter', ' emmablob',
#               ' fling93', ' iamartin', ' neelma', ' nprigoda', ' om123',
#               ' ppezz', ' rohde', ' sid', ' spiritualbeing', ' thestorygirl',
#               ' zinniz']

#     return labels[predicted_label.argmax()]


class IndexView(TemplateView):
    """
    This is the index view of the website.
    :param template_name; Specifies the static display template file.
    """
    template_name = 'index.html'


class FilesList(ListView):
    """
    ListView that display companies query list.
    :param model: Specifies the objects of which model we are listing
    :param template_name; Specifies the static display template file.
    :param context_object_name: Custom defined context object value,
                     this can override default context object value.
    """
    model = FileModel
    template_name = 'files_list.html'
    context_object_name = 'files_list'


class UploadView(CreateView):
    """
    This is the view that is used by the user of the web UI to upload a file.
    :param model: Specifies the objects of which model we are listing
    :param template_name; Specifies the static display template file.
    :param fields: Specifies the model field to be used
    :param success_url: Specifies the redirect url in case of successful upload.
    """
    model = FileModel
    fields = ['file']
    template_name = 'post_file.html'
    success_url = '/upload_success/'


class UploadSuccessView(TemplateView):
    """
    This is the success view of the UploadView class.
    :param template_name; Specifies the static display template file.
    """
    template_name = 'upload_success.html'


class SelectPredFileView(TemplateView):
    """
    This view is used to select a file from the list of files in the server.
    After the selection, it will send the file to the server.
    The server will return the predictions.
    """

    template_name = 'select_file_predictions.html'
    parser_classes = FormParser
    queryset = FileModel.objects.all()

    def get_context_data(self, **kwargs):
        """
        This function is used to render the list of files in the MEDIA_ROOT in the html template.
        """
        context = super().get_context_data(**kwargs)
        media_path = settings.MEDIA_ROOT
        myfiles = [f for f in listdir(
            media_path) if isfile(join(media_path, f))]
        context['filename'] = myfiles
        return context


class SelectFileDelView(TemplateView):
    """
    This view is used to select a file from the list of files in the server.
    After the selection, it will send the file to the server.
    The server will then delete the file.
    """
    template_name = 'select_file_deletion.html'
    parser_classes = FormParser
    queryset = FileModel.objects.all()

    def get_context_data(self, **kwargs):
        """
        This function is used to render the list of files in the MEDIA_ROOT in the html template
        and to get the pk (primary key) of each file.
        """
        context = super().get_context_data(**kwargs)
        media_path = settings.MEDIA_ROOT
        myfiles = [f for f in listdir(
            media_path) if isfile(join(media_path, f))]
        primary_key_list = []
        for value in myfiles:
            primary_key = FileModel.objects.filter(
                file=value).values_list('pk', flat=True)
            primary_key_list.append(primary_key)
        file_and_pk = zip(myfiles, primary_key_list)
        context['filename'] = file_and_pk
        return context


class FileDeleteView(views.APIView):
    """
    This class contains the method to delete a file interacting directly with the API.
    DELETE requests are accepted.
    Removing the renderer_classes an APIView instead of a TemplateView
    """
    model = FileModel
    fields = ['file']
    template_name = 'delete_success.html'
    success_url = '/delete_success/'
    renderer_classes = [TemplateHTMLRenderer]

    def post(self, request):
        """
        This method is used delete a file.
        In the identifier variable we are storing a QuerySet object.
        In the primary key object the id is extracted from the QuerySet string.
        """
        identifier = request.POST.getlist('pk').pop()
        primary_key = identifier[identifier.find("[") + 1:identifier.find("]")]
        delete_action = get_object_or_404(FileModel, pk=primary_key).delete()
        try:
            return Response({'pk': delete_action}, status=status.HTTP_200_OK)
        except ValueError as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)


class FileView(views.APIView):
    """
    This class contains the method to upload a file interacting directly with the API.
    POST requests are accepted.
    """
    parser_classes = (MultiPartParser, FormParser)
    queryset = FileModel.objects.all()

    @staticmethod
    def upload(request):
        """
        This method is used to Make POST requests to save a file in the media folder
        """
        file_serializer = FileSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            response = Response(file_serializer.data,
                                status=status.HTTP_201_CREATED)
        else:
            response = Response(file_serializer.errors,
                                status=status.HTTP_400_BAD_REQUEST)
        return response

    @staticmethod
    def check_resource_exists(file_name):
        """
        This method will receive as input the file the user wants to store
        on the server and check if a resource (an url including
        the filename as endpoint) is existing.
        If this function returns False, the user should not be able to save the
        file (or at least he/she should be prompted with a message saying that
        the file is already existing)
        """
        request = requests.get('/media/' + file_name)
        check = bool(request.status_code == 200)
        return check

    @staticmethod
    def check_file_exists(file_name):
        """
        This method will receive as input the file the user wants to store
        on the server and check if a file with this name is physically in
        the server folder.
        If this function returns False, the user should not be able to save the
        file (or at least he/she should be prompted with a message saying that
        the file is already existing)
        """
        check = bool(str(os.path.join(settings.MEDIA_ROOT, file_name)))
        return check

    @staticmethod
    def check_object_exists(file_name):
        """
        This method will receive as input the file the user wants to store
        on the server and check if an object with that name exists in the
        database.
        If this function returns False, the user should not be able to save the
        file (or at least he/she should be prompted with a message saying that
        the file is already existing)
        """
        check = FileModel.objects.get(name=file_name).exists()
        return check


class Predict(views.APIView):

    template_name = 'index.html'
    renderer_classes = [TemplateHTMLRenderer]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_name = 'speakerIdentifier.h5'

        self.loaded_model = load_model(
            os.path.join(settings.MODEL_ROOT, model_name))
        self.predictions = []
        self.graph = None

    def file_elaboration(self, filepath):

        data, sampling_rate = librosa.load(filepath, res_type='kaiser_fast')
        try:

            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate,
                                                 n_mfcc=40).T, axis=0)
            training_data = np.array(mfccs)
            training_data_expanded = training_data.reshape((1, 40))
            print(training_data)
            predicted_label = self.loaded_model.predict(training_data_expanded)
            self.graph = self.getGraph(predicted_label)

            print(predicted_label.argmax())

            numpred = predicted_label.argmax()
            self.predictions.append([self.classtospeaker(numpred)])
            print(self.predictions)
            return self.predictions
        except ValueError as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

    def post(self, request):
        filename = request.POST.getlist('file_name').pop()
        filepath = str(os.path.join(settings.MEDIA_ROOT, filename))
        predictions = self.file_elaboration(filepath)
        r = sr.Recognizer()
        text = ''
        joined_string = ''
        with sr.AudioFile(filepath) as source:
            audio_text = r.record(source)
            try:
                # using google speech recognition
                text = r.recognize_google(audio_text)
                print('Converting audio transcripts into text ...')
                print(text)
                word_tokens = word_tokenize(text)
                stop_words = set(stopwords.words('english'))
                filtered_sentence = [
                    w for w in word_tokens if not w in stop_words]
                # joined_string = ",".join(filtered_sentence)
                joined_string = ", ".join(map(str, filtered_sentence))
            except:
                print('Sorry.. run again...')

        try:
            return Response({'result': true, 'predictions': predictions.pop(), 'transscript': text, 'main_words': joined_string, 'data': self.graph}, status=status.HTTP_200_OK)
        except ValueError as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

    @staticmethod
    def classtospeaker(pred):
        label_conversion = [' A. E. Maroney', ' Andrew NG', ' Anya', ' Arielle Lipshaw',
                            ' Betty Chen', ' Bill Mosley', ' BookAngel7', ' Brendan Hodge',
                            ' Brian von Dedenroth', ' Cata', ' Christie Nowak',
                            ' David Mecionis', ' David Mix', ' Doug', ' E. Tavano', ' Hilara',
                            ' Jean Bascom', ' Jeana Wei', ' Jennifer Wiginton',
                            ' JenniferRutters', ' JenniferW', ' Jill Engle', ' John Rose',
                            ' JudyGibson', ' Julie VW', ' JustinJYN', ' Kathy Caver',
                            ' Lisa Meyers', ' M. Bertke', ' Malone', ' Mark Nelson',
                            ' Mark Welch', ' Mary J', ' Michael Packard', ' Moromis',
                            ' Nelly ()', ' Nicodemus', ' Peter Eastman', ' President Lethe',
                            ' Ransom', ' Renata', ' Russ Clough', ' S R Colon',
                            ' Scott Walter', ' Sharon Bautista', ' Simon Evers',
                            ' Stephen Kinford', ' Steven Collins', ' Susan Hooks', ' Tonia',
                            ' VOICEGUY', ' WangHaojie', ' Wayne Donovan', ' Wendy Belcher',
                            ' Winston Tharp', ' aquielisunari', ' ashleyspence', ' badey',
                            ' calystra', ' camelot2302', ' chocmuse', ' dexter', ' emmablob',
                            ' fling93', ' iamartin', ' neelma', ' nprigoda', ' om123',
                            ' ppezz', ' rohde', ' sid', ' spiritualbeing', ' thestorygirl',
                            ' zinniz']

        return label_conversion[pred]

    @staticmethod
    def getGraph(sizes):
        label_conversion = [' A. E. Maroney', ' Andrew NG', ' Anya', ' Arielle Lipshaw',
                            ' Betty Chen', ' Bill Mosley', ' BookAngel7', ' Brendan Hodge',
                            ' Brian von Dedenroth', ' Cata', ' Christie Nowak',
                            ' David Mecionis', ' David Mix', ' Doug', ' E. Tavano', ' Hilara',
                            ' Jean Bascom', ' Jeana Wei', ' Jennifer Wiginton',
                            ' JenniferRutters', ' JenniferW', ' Jill Engle', ' John Rose',
                            ' JudyGibson', ' Julie VW', ' JustinJYN', ' Kathy Caver',
                            ' Lisa Meyers', ' M. Bertke', ' Malone', ' Mark Nelson',
                            ' Mark Welch', ' Mary J', ' Michael Packard', ' Moromis',
                            ' Nelly ()', ' Nicodemus', ' Peter Eastman', ' President Lethe',
                            ' Ransom', ' Renata', ' Russ Clough', ' S R Colon',
                            ' Scott Walter', ' Sharon Bautista', ' Simon Evers',
                            ' Stephen Kinford', ' Steven Collins', ' Susan Hooks', ' Tonia',
                            ' VOICEGUY', ' WangHaojie', ' Wayne Donovan', ' Wendy Belcher',
                            ' Winston Tharp', ' aquielisunari', ' ashleyspence', ' badey',
                            ' calystra', ' camelot2302', ' chocmuse', ' dexter', ' emmablob',
                            ' fling93', ' iamartin', ' neelma', ' nprigoda', ' om123',
                            ' ppezz', ' rohde', ' sid', ' spiritualbeing', ' thestorygirl',
                            ' zinniz']
        k, = np.where(sizes[0] <= 0.005)
        print(k)

        sizes = np.delete(sizes[0], k)
        label_conversion = np.delete(label_conversion, k)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=label_conversion, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        # g = mpld3.fig_to_html(fig)
        buf = io.BytesIO()
        fig1.savefig(buf, format="png")
        buf.seek(0)
        g = base64.b64encode(buf.read())
        uri = urllib.parse.quote(g)

        return uri
