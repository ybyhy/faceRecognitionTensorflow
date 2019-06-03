import os
import numpy as np
import cv2

#指定图片的大小
IMAGE_SIZE=64

#调整图片大小
def resize_image(image,height=IMAGE_SIZE,width=IMAGE_SIZE):
    #记录数值，方便图片调整大小
    top, bottom, left, right = (0, 0, 0, 0)

    #黑色
    BLACK=[0,0,0]

    #获取图片尺寸
    h,w,_=image.shape

    #计算短边需要增加多上像素宽度使其与长边等长
    if h<w:
        dh=w-h
        top=dh//2
        bottom=dh-top
    elif w<h:
        dw=h-w
        left=dw//2
        right=dw-left
    else:
        pass

    # 给图像增加边界
    constant=cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)

    #返回图片
    return cv2.resize(constant,(height,width))

#读取训练集
images=[]
labels=[]
def read_path(path_name):
    #读取每个文件
    for dir_item in os.listdir(path_name):
        #获得路径
        full_path=os.path.abspath(os.path.join(path_name,dir_item))

        #如果是文件夹则继续调用
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            #判断文件是否为jpg文件
            if dir_item.endswith('.jpg'):
                #读取图片
                image=cv2.imread(full_path)
                #调整图片大小
                image=resize_image(image,IMAGE_SIZE,IMAGE_SIZE)

                #添加图片
                images.append(image)
                labels.append(path_name)
    return images,labels

#读取训练集数据
def load_dataset(path_name):
    images,Labels=read_path(path_name)
    #生成数组
    images=np.array(images)
    #print(images.shape)

    #标注数据
    labels=[]
    for label in Labels:
        labels.append(label[-1:])
    return images,labels


if __name__ == '__main__':
    images,labels=load_dataset("img")