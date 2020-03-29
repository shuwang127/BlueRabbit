import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Patriots.settings")
import django
django.setup()

from abandon.operation.startup import *

# 路径
excelPath = 'a_doc/data.xlsx'
importModelData(excelPath, 'place', 'Hotel', clear=True)
importModelData(excelPath, 'place', 'News', clear=True)
