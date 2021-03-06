from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.shortcuts import get_object_or_404, redirect, render
from django.core.files.storage import FileSystemStorage


# from django.template import loader

from .models import Question, Choice, dentalPreds, skinPreds

def home(request):
    return render(request, 'home.html', {})
'''
def index(request):
    qs = Question.objects.filter(pub_date__lte=timezone.now())
    context = {
        'latest_question_list': qs
    }
    return render(request, 'polls/index.html', context)
'''

def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        pred_string = dentalPreds(uploaded_file_url)
        return render(request, 'dental.html', {
            'uploaded_file_url': uploaded_file_url,
            'uploaded_file': myfile.name,
            'predictions': pred_string
        })
    return render(request, 'dental.html ')

def simple_upload_skin(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        pred_string = skinPreds(uploaded_file_url)
        return render(request, 'skin.html', {
            'uploaded_file_url': uploaded_file_url,
            'uploaded_file': myfile.name,
            'predictions': pred_string
        })
    return render(request, 'skin.html ')

class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        """
        Return the last five published questions.
        """
        # return Question.objects.order_by('-pub_date')[:5]

        """
        Return the last five published questions (not including those set to be published in the future).
        """
        # return Question.objects.filter(pub_date__lte=timezone.now()).order_by('-pub_date')[:5]

        """
            Return the last five published question (not including those set to be published in the future) with choices.
        """

        q_c = []
        qs = Question.objects.filter(pub_date__lte=timezone.now()).order_by('-pub_date')
        for q in qs:
            if q.choice_set.count() > 0:
                q_c.append(q)
        return q_c




class DetailView(generic.DetailView):
    model = Question
    template_name = 'polls/detail.html'

    def get_queryset(self):
        """
        Return the last five published questions (not including those set to be published in the future).
        """
        return Question.objects.filter(pub_date__lte=timezone.now())

class ResultsView(generic.DetailView):
    model = Question
    template_name = 'polls/results.html'


def vote(request, question_id):
     # return HttpResponse("You're voting on question %s." % question_id)
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()

        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a user hits the
        # Back button.

        return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
