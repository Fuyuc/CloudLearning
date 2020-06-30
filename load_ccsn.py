import os
from os.path import join
import cv2
import numpy as np

def img_show(img):
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load(data_dir):
    data_set = []
    count = 0
    img_label = np.zeros((1545, 1), dtype=int)
    for guy in os.listdir(data_dir):
        person_dir = join(data_dir, guy)  # 文件夹的路径
        for i in os.listdir(person_dir):
            image_dir = join(person_dir, i)  # 每个文件夹中图片的路径
            img = cv2.imread(image_dir)
            data_set.append(img)
            if guy == 'Ac':
                img_label[count] = 1
            elif guy == 'As':
                img_label[count] = 2
            elif guy == 'Cb':
                img_label[count] = 3
            elif guy == 'Cc':
                img_label[count] = 4
            elif guy == 'Ci':
                img_label[count] = 5
            elif guy == 'Cs':
                img_label[count] = 6
            elif guy == 'Ct':
                img_label[count] = 7
            elif guy == 'Cu':
                img_label[count] = 8
            elif guy == 'Ns':
                img_label[count] = 9
            elif guy == 'Sc':
                img_label[count] = 10
            elif guy == 'St':
                img_label[count] = 11
            count += 1

    data_set = np.array(data_set)
    return data_set, img_label


def img_part(img, data_set):
    part_numbers = 0
    X = Y = 0  # 起点
    PW = PH = 80  # 局部区域尺寸
    SX = SY = 10  # 步长
    while Y + PH <= img.shape[0]:
        while X + PW <= img.shape[1]:
            roi = img[Y:Y + PH, X:X + PW]
            data_set.append(roi)
            part_numbers += 1
            X = X + PW + SX
        X = 0
        Y = Y + PH + SY

    return data_set, part_numbers


if __name__ == '__main__':
    data_dir = "C:/Users/76505/Desktop/cloudset/CCSN/CCSN_v2/"
    data_set, img_label = load(data_dir)
    print(data_set.shape,img_label.shape)
    np.save('datafiles/ccsn_set.npy',data_set)
    np.save('datafiles/ccsn_label.npy',img_label)