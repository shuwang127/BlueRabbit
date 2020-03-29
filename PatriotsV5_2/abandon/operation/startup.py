import xlrd
import json
from django.apps import apps


def read_excel_to_diclist(filePath, sheetName):
    try:
        table = xlrd.open_workbook(filePath).sheet_by_name(sheetName)
        rows = table.nrows
        col_names = table.row_values(0)
        data = []
        for row_num in range(1, rows):
            row_value = table.row_values(row_num)
            if row_value:
                app = {}
                for i in range(len(col_names)):
                    app[col_names[i]] = row_value[i]
                data.append(app)
        return data
    except Exception as e:
        return None


def importModelData(excelPath, appName, modelName, clear=False):
    modelObj = apps.get_model(appName, modelName)
    sheetname = modelName
    print('import model: %s ...' % modelName)
    if clear:
        modelObj.objects.all().delete()
    data = read_excel_to_diclist(excelPath, sheetname)
    for item in data:
        modelObj(**json.loads(json.dumps(item))).save()
    print('import model: %s done!' % modelName)


# 导入数据 示例
# import os
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Patriots.settings")
# import django
# django.setup()
#
# from abandon.operation.startup import *
#
# # 路径
# excelPath = 'a_doc/data.xlsx'
# importModelData(excelPath, 'place', 'Hotel')
