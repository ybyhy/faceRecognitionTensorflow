#卷积层和池化层
#卷积类似于图像处理中的特征提取操作，池化则很类似于降维,常用的有最大池化和平均池化
from keras.layers import Conv2D,MaxPooling2D

#引入全连接层、Dropout、Flatten
#全连接层就是经典的神经网络全连接。
#Dropout用来在训练时按一定概率随机丢弃一些神经元，以获得更高的训练速度以及防止过拟合。
#Flatten用于卷积层与全连接层之间，把卷积输出的多维数据拍扁成一维数据送进全连接层（类似shape方法）
from keras.layers import Dense,Dropout,Flatten

#引入SGD（梯度下降优化器）来使损失函数最小化
from keras.optimizers import SGD
from sklearn.model_selection import  train_test_split
import numpy as np
import keras
from keras.utils import np_utils

from algorithm.加载数据集 import load_dataset,IMAGE_SIZE

#读入所有图像和标签
raw_images,raw_labels=load_dataset('img')
#把图像转换为float类型，方便归一化
#raw_images,raw_labels=np.asarray(raw_images,dtype=np.float32),np.asarray(raw_labels,dtype=np.float32)
raw_images=np.asarray(raw_images,dtype=np.float32)
raw_labels=np.asarray(raw_labels,dtype=np.float32)

#使所有标签平等化(on-hot编码)
one_hot_labels=np_utils.to_categorical(raw_labels)

#划分训练集和测试集(7:3)
x_train,x_test,y_train,y_test=train_test_split(raw_images,one_hot_labels,test_size=0.3)

#数据归一化，图像数据只需要每个像素除以255就可以
x_train=x_train/255.0
x_test=x_test/255.0

#构建卷积神经网络的每一层

#卷积层
##序贯模型,为最简单的线性、从头到尾的结构顺序，不分叉，是多个网络层的线性堆叠
face_recognition_model=keras.Sequential()
                                  #卷积核数量：32，大小：3*3
face_recognition_model.add(Conv2D(32,3,3,
                                  #边缘不补充
                                  border_mode='valid',
                                  #卷积步长：1
                                  subsample=(1,1),
                                  #使用tf运算
                                  dim_ordering='tf',
                                  #图片尺寸
                                  input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
                                  #激活函数：relu(收敛速度较快)
                                  activation='relu'))
face_recognition_model.add(Conv2D(32,3,3,border_mode='valid',subsample=(1,1),dim_ordering='tf',activation='relu'))

#池化层，过滤器尺寸:2 * 2
face_recognition_model.add(MaxPooling2D(pool_size=(2,2)))

#Dropout层,防止过拟合(对于神经网络单元，暂时随机丢弃)
face_recognition_model.add(Dropout(0.2))

face_recognition_model.add(Conv2D(64,3,3,border_mode='valid',subsample=(1,1),dim_ordering='tf',activation='relu'))
face_recognition_model.add(Conv2D(64,3,3,border_mode='valid',subsample=(1,1),dim_ordering='tf',activation='relu'))

face_recognition_model.add(MaxPooling2D(pool_size=(2,2)))

face_recognition_model.add(Dropout(0.2))

#Flatten层将图片的卷积输出压扁成一个一维向量
face_recognition_model.add(Flatten())

#Dense层（全连接层）512个神经元
face_recognition_model.add(Dense(512,activation='relu'))

face_recognition_model.add(Dropout(0.4))

#输出层，神经元数是标签种类数，使用sigmoid激活函数
face_recognition_model.add(Dense(len(one_hot_labels[0]),activation='sigmoid'))

#打印神经网络结构
face_recognition_model.summary()

#优化：SGD（随机梯度优化，对每个训练样本进行参数更新，每次执行都进行一次更新，且执行速度更快）
                  #学习率：0.01
sgd_optimizer=SGD(lr=0.01,
                  #学习率衰减值：1e-6
                  decay=1e-6,
                  #动量参数:0.8
                  momentum=0.8,nesterov=True)

#编译模型
##损失函数
face_recognition_model.compile(loss='categorical_crossentropy',optimizer=sgd_optimizer,metrics=['accuracy'])

#开始训练
##训练次数：100
face_recognition_model.fit(x_train,y_train,epochs=25,batch_size=300,shuffle=True,validation_data=(x_test,y_test))

#评估结果并保存模型
print(face_recognition_model.evaluate(x_test,y_test,verbose=0))
face_recognition_model.save('model/model.h5')