import cv2
from algorithm.Model import Model
from algorithm.加载数据集 import IMAGE_SIZE,resize_image
from faceRecognitionTensorflow import settings

def recognition_image(path):
    path=path.replace("\\","\\\\")
    #存储名字
    name=''
    #加载模型
    model = Model()
    model.load_model(file_path='E:\\PyCharm\\porject\\faceRecognitionTensorflow\\algorithm\\model\\model.h5')

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 人脸识别分类器本地存储路径
    cascade_path = "E:\\PyCharm\\porject\\faceRecognitionTensorflow\\algorithm\\file\\haarcascade_frontalface_alt2.xml"

    #获取图片
    img=cv2.imread(path)

    # 图像灰化，降计算复杂度
    img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 使用人脸识别分类器，读入分类器
    cascade = cv2.CascadeClassifier(cascade_path)

    # 利用分类器识别出哪个区域为人脸
    faceRects = cascade.detectMultiScale(img_grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect

            # 截取脸部图像提交给模型识别这是谁
            image = img[y - 10: y + h + 10, x - 10: x + w + 10]
            faceID = model.face_predict(image, IMAGE_SIZE)
            #print(faceID)

            # 如果是“羊泓运”
            if faceID == 0:
                name='yhy'
                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                # 文字提示是谁
                cv2.putText(img, 'yhy',
                            # 坐标
                            (x +30, y -30),
                            # 字体
                            cv2.FONT_HERSHEY_SIMPLEX,
                            # 字号
                            1,
                            # 颜色
                            (255, 0, 255),
                            # 字的线宽
                            2)
            # 如果是“陈泽茂”
            elif faceID == 1:
                name='czm'
                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                # 文字提示是谁
                cv2.putText(img, 'czm',
                            (x + 30, y - 30),  # 坐标
                            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                            1,  # 字号
                            (255, 0, 255),  # 颜色
                            2)  # 字的线宽
            # 如果是“龙嘉鑫”
            elif faceID == 2:
                name='ljx'
                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                # 文字提示是谁
                cv2.putText(img, 'ljx',
                            (x + 30, y - 30),  # 坐标
                            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                            1,  # 字号
                            (255, 0, 255),  # 颜色
                            2)  # 字的线宽
            else:
                pass

    cv2.imwrite('E:/PyCharm/porject/faceRecognitionTensorflow/static/media/images_recognition/'+name+'.jpg',img)
    #url='faceRecognitionTensorflow\\static\\media\\images_recognition\\'+name+'.jpg'
    url = '\\static\\media\\images_recognition\\' + name + '.jpg'
    #显示图片
    #cv2.imshow("img",img)
    #等待图片关闭
    #cv2.waitKey()
    return url

def test(img):
    cv2.imshow("img",img)
    cv2.waitKey()


if __name__=='__main__':
    #img = recognition_image('file/yhy.jpg')
    #img = recognition_image('../static/media/images/yhy.jpg')
    img=recognition_image(r'E:\PyCharm\porject\faceRecognitionTensorflow\static\media\images\yhy.jpg')
    #test(img)