from django.contrib import admin
from .models import *


@admin.register(Hotel)
class HotelAdmin(admin.ModelAdmin):
    list_display = ['name', 'zipCode', 'location', 'type', 'lat', 'lon', 'risk']

# @admin.register(Panel)
# class PanelAdmin(admin.ModelAdmin):
#     list_display = ['title', 'sort']


@admin.register(News)
class NewsAdmin(admin.ModelAdmin):
    list_display = ['title',]
