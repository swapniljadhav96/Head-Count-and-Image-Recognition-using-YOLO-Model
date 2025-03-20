"""
URL configuration for HeadCountProject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from home import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name = "home"),
    path("login/", views.login , name = "login"),
    path("create/", views.create , name = "create"),
    path("failed/", views.failed , name = "failed"),
    path("logout/", views.logout , name = "logout"),
    path("logout_2/", views.logout_2 , name = "logout_2"),
    path("startdetection/", views.startdetection , name = "startdetection"),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)