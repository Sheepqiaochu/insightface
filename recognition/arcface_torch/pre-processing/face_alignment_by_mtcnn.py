# -*- coding: utf-8 -*-
# @Time    : 2019/7/26 13:54
# @Author  : "梅俊辉"
# @Email   : 18211091722@163.com
# @File    : make_images_by_mtcnn.py 
# @Software: PyCharm


import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../..', 'src', 'common'))
import face_image
import face_preprocess
import numpy as np

mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')

detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

faces_path = r'F:\datasets\20_faces'          # 人脸数据文件夹
output_path = r'F:\datasets\20_faces_clip'    # 对齐后的保存的人脸数据文件夹

for root,_,files in os.walk(faces_path):
    for fname in files:
        img = cv2.imread(os.path.join(root,fname))
        # new_root = root.replace('20_faces','20_faces_clip')
        new_root = os.path.join(output_path,os.path.basename(root))
        if not os.path.exists(new_root):
            os.mkdir(new_root)
        # run detector
        results = detector.detect_face(img)

        if results is None:
            continue
        bbox, points = results
        if bbox.shape[0] == 0:
            continue
        bbox = bbox[0, 0:4]
        points = points[0, :].reshape((2, 5)).T
        # print(bbox)
        # print(points)
        nimg = face_preprocess.preprocess(img, bbox, points, image_size='112,112')
        # aligned = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        # aligned = np.transpose(nimg, (2, 0, 1))

        aligned = np.transpose(nimg, (0, 1, 2))
        # cv2.imshow('aligned', aligned)
        cv2.imwrite(os.path.join(new_root, fname), aligned)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)