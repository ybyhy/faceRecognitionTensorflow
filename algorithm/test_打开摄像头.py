# -*- coding:utf8 -*-
import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)
i=1
while True:
    # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
    hx, frame = cap.read()
    if hx is False:
        print('read video error')
        exit()

    # 窗口设置为自动调节大小
    cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite('file/'+str(i)+'.jpg',frame)
        break
    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放摄像头
cap.release()

# 结束所有窗口
cv2.destroyAllWindows()