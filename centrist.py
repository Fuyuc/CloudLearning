import cv2
import numpy as np

def caculate_binary(list):
    sum = 0
    for i in range(len(list)):
        sum += list[i]*pow(2,len(list)-i-1)
    return sum

def centrist(img,X,Y):
    cen_feat = []
    i , j = X , Y
    center = img[X+1,Y+1]
    while i < X + 3:
        while j < Y + 3:
            cen_feat.append(0 if img[i,j] > center else 1)
            j += 1
        j = Y
        i += 1
    del cen_feat[4]
    return cen_feat

def extract_cent(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    centrist_feat = []
    X = Y = 0
    count = 0
    while X + 3 <= gray.shape[0]:
        while Y + 3 <= gray.shape[1]:
            centrist_feat.append(caculate_binary(centrist(gray,X,Y)))
            count += 1
            Y += 1
        Y = 0
        X += 1
    centrist_feat = np.array(centrist_feat,dtype=np.uint8)
    his_cent = cv2.calcHist([centrist_feat], [0], None, [256], [0, 255])
    return his_cent.flatten()


if __name__ == '__main__':
    data_set = np.load('data_set.npy', allow_pickle=True)
    img_label = np.load('img_label.npy', allow_pickle=True)
    print(extract_cent(data_set[0]))