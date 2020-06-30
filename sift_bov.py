import os
from os.path import join
import cv2
import numpy as np
from model_selection import train_test_split
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
import load_data
from SVM import SVM
from fishervector import FisherVectorGMM

# 指定云类型
Cloud_category = {
    "1": "卷云",
	"2": "卷层云",
	"3": "卷积云",
	"4": "高积云",
    "5": "高层云",
    "6": "层积云",
    "7": "层云",
    "8": "雨层云",
    "9": "积云",
    "10": "积雨云",
    "11": "其他",
}

def extract_sift(img_set,img_label):
    feat_vec = []    #存储sift向量集
    y = []      #标签集
    for i,img in enumerate(img_set):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray, None)  #提取sift特征点及其特征
        if len(kp) >= 200:
            y.append(img_label[i])
            # img = cv2.drawKeypoints(gray, kp, img)
            # img_show(X[i])
            des = np.array(np.random.permutation(des[:50]),dtype=int).reshape(1,-1)
            # model = KMeans(n_clusters=3, n_jobs=4, max_iter=500)  # 分为k类, 并发数4    #89-91，94-97使用Kmeans/BOW方法生成直方图向量
            # model.fit(features)  # 开始聚类
            # label_list.append(model.labels_)
            feat_vec.append(des[0])
    # vector = np.zeros((len(label_list), 8), dtype=int)
    # for i,label in enumerate(label_list):
    #     for bin in label:
    #         vector[i][bin] = vector[i][bin]+1
    y = np.array(y,dtype=int).reshape(1,len(y))
    return feat_vec,y


if __name__ == '__main__':
    data_dir = "C:/Users/76505/Desktop/cloudimgs/"
    data_set, img_label = load_data.load(data_dir)
    print(data_set.shape,img_label.shape)
    img_set = load_data.resize(data_set)
    feature_vec,y = extract_sift(img_set, img_label)
    print(y)
    print(SVM(feature_vec,y))