"""Patriots URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from place.views import *
from django.views.generic import RedirectView
from django.conf.urls import url
from django.views.static import serve
from django.conf import settings


urlpatterns = [
    path('', RedirectView.as_view(url='/dashboard/')),
    path('background/', background),
    path('dashboard/', dashboard),
    path('analysis/', analysis),
    path('reference/', reference),
    path('admin/', admin.site.urls),
    url(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),

]
