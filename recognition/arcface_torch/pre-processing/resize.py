import os

import cv2
from PIL import Image

# faces_path = r'D:\yangqiancheng\Desktop\images'  # 人脸数据文件夹
# output_path = r'D:\yangqiancheng\Desktop\cropped'  # 对齐后的保存的人脸数据文件夹

faces_path = r'~/datasets/images'  # 人脸数据文件夹
output_path = r'~/datasets/oppo'  # 对齐后的保存的人脸数据文件夹

for root, _, files in os.walk(faces_path):
    for fname in files:
        img = cv2.imread(os.path.join(root, fname))
        # new_root = root.replace('20_faces','20_faces_clip')
        new_root = os.path.join(output_path, os.path.basename(root))
        if not os.path.exists(new_root):
            os.mkdir(new_root)
        cropped = img[4:116, 4:116]
        cv2.imwrite(os.path.join(new_root, fname), cropped)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
