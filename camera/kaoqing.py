import cv2
from cv2 import face
import os
import numpy as np
from PIL import Image
import datetime
import csv


# 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
Path = r"D:\opencv-4.4.0\data\haarcascades\haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(Path)
names = ['name1', 'name2', '', '', '', '', '' '','','','','','','','','','','','','','','','','','','name23', 'name24']
zh_name = ['我', '你']

'''
with open("maxmember.csv", "r", encoding='UTF-8') as csv_file:
    reader = csv.reader(csv_file)
    for item in reader:
        # print(item)
        names.append(item[2])
        zh_name.append(item[1])
    # print (zh_name)
'''


def data_collection():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW是作为open调用的一部分传递标志，还有许多其它的参数，而这个CAP_DSHOW是微软特有的。
    face_id = input('\n 请输入你的ID:')
    print('\n 数据初始化中，请直视摄像机录入数据....')
    count = 0
    while cap.isOpened():
        # 从摄像头读取图片
        sucess, img = cap.read()
        # 转为灰度图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        # 1.image表示的是要检测的输入图像# 2.objects表示检测到的人脸目标序列# 3.scaleFactor表示每次图像尺寸减小的比例
        for (x, y, w, h) in faces:
            # 画矩形
            cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 0))
            count += 1
            # 保存图像
            cv2.imwrite("facedata/Member." + str(face_id) + '.' + str(count) + '.jpg', gray[y: y + h, x: x + w])
        cv2.imshow('data collection', img)
        # 保持画面的持续。
        k = cv2.waitKey(10)
        if k == 27:  # 通过esc键退出摄像
            break
        elif count >= 200:  # 得到n个样本后退出摄像
            break
    cap.release()
    cv2.destroyAllWindows()


def face_training():
    # 人脸数据路径
    path1 = './facedata'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print('数据训练中')
    faces, ids = get_images_and_labels(path1)
    recognizer.train(faces, np.array(ids))
    recognizer.write(r'.\trainer.yml')


# LBP是一种特征提取方式，能提取出图像的局部的纹理特征
def get_images_and_labels(path):
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]  # join函数将多个路径组合后返回
    print(imagepaths)
    facesamples = []
    ids = []
    # 遍历图片路径，导入图片和id，添加到list
    for imagePath in imagepaths:
        pil_img = Image.open(imagePath).convert('L')  # 通过图片路径并将其转换为灰度图片。
        img_numpy = np.array(pil_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            facesamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)
    return facesamples, ids


def face_ientification():

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./trainer.yml')
    faceCascade = cv2.CascadeClassifier(Path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    idnum = 0
    global names
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # 设置大小
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        # 图像灰度处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,  # 表示在前后两次相继的扫描中，搜索窗口的比例系数
            minNeighbors=5,  # 表示构成检测目标的相邻矩形的最小个数(默认为3个)
            minSize=(int(minW), int(minH))  # minSize和maxSize用来限制得到的目标区域的范围
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 返回侦测到的人脸的id和近似度conf（数字越大和训练数据越不像）
            idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 100:
                namess = names[idnum]
                confidence = "{0}%".format(round(100 - confidence))
            else:
                namess = "unknown"
                confidence = "{0}%".format(round(100 - confidence))

            cv2.putText(img, str(namess), (x + 5, y - 5), font, 1, (0, 0, 255), 1)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (0, 0, 0), 1)  # 输出置信度

        cv2.imshow(u'Identification punch', img)
        k = cv2.waitKey(5)
        if k == 13:
            theTime = datetime.datetime.now()
            # print(zh_name[idnum])
            strings = [str(zh_name[idnum]), str(theTime)]
            print(strings)
            with open("log.csv", "a", newline="") as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([str(zh_name[idnum]), str(theTime)])
        elif k == 27:
            cam.release()
            cv2.destroyAllWindows()
            break


while True:
    a = int(input("输入1，录入脸部，输入2进行识别打卡:"))
    if a == 1:
        data_collection()
    elif a == 2:
        face_ientification()
    elif a == 3:
        face_training()
