import numpy as np
import cv2
import load_data
import cnn_feature
from centrist import extract_cent
import fisher
from fisher import fisher_vector
from SVM import SVM
from sift import sift_descripter
from color_histogram import extract_his
import warnings

# 存储所有图片特征融合的特征集集 shape(n,128)，用于EM预估参数
def extract_descripter(data_set,size,stride):
    all_feat_list = np.zeros((1,156),dtype=np.float32)
    single_img_sift_list = []  # 存储每张图片分离的特征集，shape(n,m,128)
    y = []  # 标签集
    for i,img in enumerate(data_set):
        one_img_feat = np.zeros((1,156),dtype=np.float32)
        X = Y = 0  # 起点
        PW = PH = size  # 局部区域尺寸
        SX = SY = stride  # 步长
        while X + PH <= img.shape[0]:
            while Y + PW <= img.shape[1]:
                roi = img[X:X + PH, Y:Y + PW]
                kp, des_list = sift_descripter(roi)
                if len(kp):
                    his_feat = extract_his(roi)
                    # cent_feat = extract_cent(roi)
                    part_img_feat = np.array([np.hstack((des,his_feat)) for des in des_list])
                    all_feat_list = np.concatenate((all_feat_list, part_img_feat), axis=0)
                    one_img_feat = np.concatenate((one_img_feat, part_img_feat), axis=0)
                Y = Y + PW + SY
            Y = 0
            X = X + PH + SX
        one_img_feat = np.delete(one_img_feat, 0, 0)
        single_img_sift_list.append(one_img_feat)
    all_feat_list = np.delete(all_feat_list, 0, 0)
    return all_feat_list,np.array(single_img_sift_list)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    data_set = np.load('datafiles\ccsn_set.npy',allow_pickle=True)
    img_label = np.load('datafiles\ccsn_label.npy',allow_pickle=True)

    # cnn_feature.trainandsave()
    # np.save('cnn_list.npy',cnn_feature.test())
    # cnn_list = np.load('cnn_list.npy',allow_pickle=True)


    feat_list, single_img_sift_list = extract_descripter(data_set, 256, 0)
    #除去无SIFT描述符的图片
    remove_ids = []
    for i, feat in enumerate(single_img_sift_list):
        if feat.shape[0] == 0:
            remove_ids.append(i)
    remove_ids = np.array(remove_ids)
    single_img_sift_list = np.delete(single_img_sift_list, remove_ids, 0)
    img_label = np.delete(img_label, remove_ids, 0)
    # cnn_list = np.delete(cnn_list, remove_ids, 0)

    gmm_feat = fisher.generate_gmm(feat_list, 5)
    feat_list = np.array([fisher_vector(img_feat, *gmm_feat) for img_feat in single_img_sift_list])

    # cnn_his = fisher.generate_gmm(cnn_list, 5)
    # cnn_feat_list = np.array([fisher.fisher_vector(img_feat,*cnn_his) for img_feat in cnn_list])
    # feat_list = np.concatenate((feat_list,cnn_feat_list),axis=1)

    print(feat_list.shape,img_label.shape)
    print(SVM(feat_list, img_label))