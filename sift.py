import cv2
from load_data import img_show
import numpy as np
import load_data
from SVM import SVM
from fishervector import FisherVectorGMM

def sift_descripter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, _ = sift.detectAndCompute(gray, None)
    # img = cv2.drawKeypoints(gray, kp, img)
    # img_show(img)
    return sift.detectAndCompute(gray, None) # 提取sift特征点及其特征