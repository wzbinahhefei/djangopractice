import json
import random

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


def tank4c9(request):
    assert isinstance(request, HttpRequest)
    return render(request, 'collector/tank4c9.html',)


def gettank4c9data(request):

    tank4c9 = {
        'Status':  random.randint(0, 1),  # 设备运行状态
        'OverheadFlow': random.randint(1, 10),  # '顶流量',
        'ButtomsFlow': random.randint(1, 10),  # '低流量'
        'Power':  random.randint(10000, 100000),  # 功率
    }
    return HttpResponse(json.dumps(tank4c9))


def getcollectordata(request):

    tank4c9 = {
        'DeviceId': 1,
        'DeviceName': '1#反应罐',
        'Status': 1,  # 设备运行状态
        'OverheadFlow': 0,  # '顶流量',
        'ButtomsFlow': 0,  # '低流量'
        'Power': 0,  # 功率
    }

    import OpenOPC
    opc = OpenOPC.client()
    opc.connect('Matrikon.OPC.Simulation')
    tank4c9['OverheadFlow'] = opc['Random.Int1']
    tank4c9['ButtomsFlow'] = opc['Random.Int2']
    tank4c9['Power'] = opc['Random.Int4']
    opc.close()

    collector = {
         'CollectorId': 1,
         'CollectorName': '1#采集器',
         'Status': 0,
         'DeviceList': [tank4c9],
         }

    return HttpResponse(json.dumps(collector))
