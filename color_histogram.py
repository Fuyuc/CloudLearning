import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import load_data

def extract_his(img):
    his_feat = np.zeros((2,14),dtype=np.float32)
    def oppose_space(img):
        oppose_img = np.zeros(img.shape,img.dtype)
        oppose_img[0] = (img[2] - img[0])/ math.sqrt(2)
        oppose_img[1] = (img[2] + img[1]-2*img[0])/ math.sqrt(6)
        oppose_img[2] = (img[0] + img[1] + img[2])/ math.sqrt(3)
        return oppose_img

    def R_B_differ(img):
        rb_differ_img = img[2] - img[0]
        return np.array(rb_differ_img,dtype=np.uint8)

    def R_B_specfic(img):
        try:
            rb_specifc_img = img[2] / img[0]
        except:
            print("除数存在0元素")
        rb_specifc_img[rb_specifc_img > 1000] = 0
        return np.array(rb_specifc_img,dtype=np.uint8)   #直方图输入格式统一uint8

    mean, std = cv2.meanStdDev(img)


    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv,std_hsv = cv2.meanStdDev(img_hsv)
    mean = np.concatenate((mean,mean_hsv),axis=0)
    std = np.concatenate((std,std_hsv), axis=0)


    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mean_lab, std_lab = cv2.meanStdDev(img_lab)
    mean = np.concatenate((mean, mean_lab), axis=0)
    std = np.concatenate((std, std_lab), axis=0)

    img_oppose = oppose_space(img)
    mean_oppose, std_oppse = cv2.meanStdDev(img_oppose)
    mean = np.concatenate((mean, mean_oppose), axis=0)
    std = np.concatenate((std, std_oppse), axis=0)

    mean_differ, std_differ = cv2.meanStdDev(R_B_differ(img))
    mean = np.concatenate((mean, mean_differ), axis=0)
    std = np.concatenate((std, std_differ), axis=0)

    mean_specfic, std_specfic = cv2.meanStdDev(R_B_specfic(img))
    mean = np.concatenate((mean, mean_specfic), axis=0)
    std = np.concatenate((std, std_specfic), axis=0)

    his_feat = np.concatenate((mean,std))
    return his_feat.reshape(1,-1)[0]

    # color = ('b','g','r')
    # img_histr_bgr = []
    # for i,col in enumerate(color):
    #     histr = cv2.calcHist([img],[i],None,[10],[0,256])
    #     img_histr_bgr.append(histr)
    # #     plt.plot(histr,color = col)
    # #     plt.xlim([0,26])
    # # plt.show()
    #
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # for i,col in enumerate(color):
    #     histr = cv2.calcHist([img_hsv],[i],None,[10],[0,256])
    #     img_histr_bgr.append(histr)
    #
    # img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # for i,col in enumerate(color):
    #     histr = cv2.calcHist([img_lab],[i],None,[10],[0,256])
    #     img_histr_bgr.append(histr)
    #
    #
    # for i,col in enumerate(color):
    #     histr = cv2.calcHist([oppose_space(img)],[i],None,[10],[0,256])
    #     img_histr_bgr.append(histr)
    #
    #
    # histr = cv2.calcHist([R_B_differ(img)],[0],None,[10],[0,256])
    # img_histr_bgr.append(histr)
    #
    #
    # # def meanandvariance():
    #
    # histr = cv2.calcHist([R_B_specfic(img)],[0],None,[10],[0,26])
    # img_histr_bgr.append(histr)
    # return np.array(img_histr_bgr)



if __name__ == '__main__':
    data_dir = "C:/Users/76505/Desktop/cloudimgs/train/"
    data_set, img_label = load_data.load(data_dir)
    print(extract_his(data_set[0]))