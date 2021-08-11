# -*- coding: utf-8 -*-
# @Time    : 2019/7/29 16:19
# @Author  : "梅俊辉"
# @Email   : 18211091722@163.com
# @File    : write_pairs.py 
# @Software: PyCharm

import os
import itertools

def txt_writer(path,txt_name):
    faces = []
    for root,_,files in os.walk(path):
        for fname in files:
            suffix = os.path.splitext(fname)[1].lower()
            if suffix in ['.jpg','.jpeg']:
                faces.append(os.path.join(root,fname))

    iter = list(itertools.combinations(faces, 2))
    for idx,(f1,f2) in enumerate(iter):
        if os.path.basename(os.path.split(f1)[0]) == \
                os.path.basename(os.path.split(f2)[0]):
            iter[idx] = [f1,f2,1]
        else:
            iter[idx] = [f1, f2, 0]
    with open(txt_name,'w') as txt:
        # txt.write(json.dumps(iter))
        for data in iter:
            txt.write(data[0]+','+data[1]+','+str(data[2])+'\n')
    return iter

if __name__ == "__main__":
    txt_writer(r'F:\datasets\20_faces_clip',#数据目录
               r'F:\datasets\20_faces_datasets\pairs.txt'#生成文件目录
               )