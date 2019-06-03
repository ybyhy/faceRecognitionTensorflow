
from django.shortcuts import render
from algorithm.face_recognition import face_recognition
from algorithm.image_recognition import recognition_image
from django.views.decorators.csrf import csrf_exempt
import  json
import os
import cv2
from . import settings

def index(request):
    return render(request,'人脸识别.html')

def image(request):
    return render(request,'图片识别.html')

def video(requset):
    return render(requset,'视频识别.html')

def video_recognition(request):
    face_recognition()
    return render(request,'视频识别.html')

@csrf_exempt
def image_recognition(request):
    #获取图片
    img=request.FILES.get("img")
    #存储图片到静态地址
    url=settings.MEDIA_ROOT+"\\images\\"+img.name
    with open(url, 'wb') as f:
        #循环读取图片内容，每次只从本地磁盘读取一部分图片内容，加载到内存中，并将这一部分内容写入到目录下，写完以后，内存清空；下一次再从本地磁盘读取一部分数据放入内存。就是为了节省内存空间。
        #pic.chunks()
        for data in img.chunks():
            f.write(data)
    context={}
    context['image_recognition_url']=recognition_image(url)
    print(context['image_recognition_url'])
    return render(request,'识别结果.html',{'context':json.dumps(context)})