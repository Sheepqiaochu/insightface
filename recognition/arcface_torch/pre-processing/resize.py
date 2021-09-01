import os
import numpy
import cv2
from PIL import Image
from torchvision import transforms

# faces_path = r'/data/users/yangqiancheng/datasets\images'  # 人脸数据文件夹
# output_path = r'/data/users/yangqiancheng/datasets\cropped'  # 对齐后的保存的人脸数据文件夹

faces_path = r'/data/users/yangqiancheng/datasets/images'  # 人脸数据文件夹
output_path = r'/data/users/yangqiancheng/datasets/cropped'  # 对齐后的保存的人脸数据文件夹


def resize_for_training():
    for root, _, files in os.walk(faces_path):
        for fname in files:
            img = Image.open(os.path.join(root, fname))
            # new_root = root.replace('20_faces','20_faces_clip')
            new_root = os.path.join(output_path, os.path.basename(root))
            if not os.path.exists(new_root):
                os.mkdir(new_root)
            cropped = transforms.Compose([
                transforms.CenterCrop(112)])
            # cropped_image = cv2.cvtColor(numpy.asarray(cropped(img)), cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(new_root, fname), cropped_image)
            cropped_image = cropped(img)
            cropped_image.save(os.path.join(new_root, fname))
            # cv2.imshow("img", img)
            # cv2.waitKey(0)


if __name__ == '__main__':
    resize_for_training()
