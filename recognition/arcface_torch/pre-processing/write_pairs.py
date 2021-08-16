# -*- coding: utf-8 -*-
# @Time    : 2019/7/29 16:19
# @Author  : "梅俊辉"
# @Email   : 18211091722@163.com
# @File    : write_pairs.py 
# @Software: PyCharm

import os
import itertools


def txt_writer(path, txt_name):
    faces = []
    for root, _, files in os.walk(path):
        for fname in files:
            suffix = os.path.splitext(fname)[1].lower()
            if suffix in ['.jpg', '.jpeg']:
                faces.append(os.path.join(root, fname))

    iter = list(itertools.combinations(faces, 2))
    for idx, (f1, f2) in enumerate(iter):
        if os.path.basename(os.path.split(f1)[0]) == \
                os.path.basename(os.path.split(f2)[0]):
            iter[idx] = [f1, f2, 1]
        else:
            iter[idx] = [f1, f2, 0]
    with open(txt_name, 'w') as txt:
        # txt.write(json.dumps(iter))
        for data in iter:
            txt.write(data[0] + ',' + data[1] + ',' + str(data[2]) + '\n')
    return iter


def read_names(path, txt_name):
    faces = []
    with open(path, 'r') as test_list:
        for lines in test_list:
            faces.append(lines.split("\t")[1])
    iter = list(itertools.combinations(faces, 2))
    for idx, (f1, f2) in enumerate(iter):
        if os.path.basename(os.path.split(f1)[0]) == \
                os.path.basename(os.path.split(f2)[0]):
            iter[idx] = [f1, f2, 1]
        else:
            iter[idx] = [f1, f2, 0]
    with open(txt_name, 'w') as txt:
        # txt.write(json.dumps(iter))
        for data in iter:
            txt.write(data[0] + ',' + data[1] + ',' + str(data[2]) + '\n')
    return iter


if __name__ == "__main__":
    read_names(r'/data/users/yangqiancheng/datasets/oppo_cropped/test.lst',  # path of test.lst
               r'/data/users/yangqiancheng/datasets/oppo_cropped/pairs.txt'  # path of saving pairs.txt
               )
