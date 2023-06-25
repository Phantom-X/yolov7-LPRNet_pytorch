import os
import random
from shutil import copy2

imgpath = ["/home/phantom/Projects/PythonProjects/yolov7-LPRNet_pytorch/VOCdevkit/CCPD2019/CCPD2019/ccpd_base",
           "/home/phantom/Projects/PythonProjects/yolov7-LPRNet_pytorch/VOCdevkit/CCPD2019/CCPD2019/ccpd_blur",
           "/home/phantom/Projects/PythonProjects/yolov7-LPRNet_pytorch/VOCdevkit/CCPD2019/CCPD2019/ccpd_challenge",
           "/home/phantom/Projects/PythonProjects/yolov7-LPRNet_pytorch/VOCdevkit/CCPD2019/CCPD2019/ccpd_db",
           "/home/phantom/Projects/PythonProjects/yolov7-LPRNet_pytorch/VOCdevkit/CCPD2019/CCPD2019/ccpd_fn"]

for p in imgpath:
    files = os.listdir(p)
    num_files = len(files)
    print("num_files: " + str(num_files))
    index_list = list(range(num_files))
    random.shuffle(index_list)  # 打乱顺序
    SaveDir = r"/home/phantom/Projects/PythonProjects/yolov7-LPRNet_pytorch/VOCdevkit/VOC2007/JPEGImages"
    if p == "/home/phantom/Projects/PythonProjects/yolov7-LPRNet_pytorch/VOCdevkit/CCPD2019/CCPD2019/ccpd_base":
        for i in index_list[:4000]:
            fileName = os.path.join(p, files[i])  # （图片文件夹）+图片名=图片地址
            copy2(fileName, SaveDir)
    else:
        for i in index_list[:2000]:
            fileName = os.path.join(p, files[i])  # （图片文件夹）+图片名=图片地址
            copy2(fileName, SaveDir)
