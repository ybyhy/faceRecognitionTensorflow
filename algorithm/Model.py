from keras.models import load_model
from keras import backend
from algorithm.加载数据集 import resize_image

class Model:
    def __init__(self):
        self.model = None

    def load_model(self, file_path):
        self.model = load_model(file_path)

    # 识别人脸
    def face_predict(self, image,IMAGE_SIZE):
        #根据后端系统确定维度顺序
        if backend.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = resize_image(image)
            # 与模型训练不同，这次只是针对1张图片进行预测
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        elif backend.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        # 浮点并归一化
        image = image.astype('float32')
        image =image/255

        # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
        result = self.model.predict_proba(image)
        #print('result:', result)

        # 给出类别预测：0或者1
        result = self.model.predict_classes(image)

        # 返回类别预测结果
        return result[0]