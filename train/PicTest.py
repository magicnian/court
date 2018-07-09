import os
import cv2
import time
import numpy as np

# from keras.preprocessing import image

CAPTCHA_DIR = "E:\\document\\ocr\\court\\zhixing\\dst\\"

width = 160
height = 70


def test_cv2():
    # a = [[[2], [3]], [[4], [5]]]
    # n_a = np.array(a)
    # print(n_a.shape)
    pic_path = "E:\\document\\ocr\\court\\zhixing\\soruce\\2ape.png"
    img = cv2.imread(pic_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = img_gray.reshape((70, 160, 1))
    # cv2.imwrite('test.png', img_gray)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
    cv2.imwrite('test2.png', thresh)
    print(thresh.shape)


def pic_deal():
    '''
    使用cv2.imread函数，将flags参数设置为0，
    将以灰度模式读取一张图片，导致读出的array的shape为(height,width)
    可以通过reshape转换为(height,width,1)
    :return:
    '''
    img_names = os.listdir(CAPTCHA_DIR)
    for name in img_names:
        # raw_img = image.load_img(CAPTCHA_DIR + name, target_size=(height, width))
        # img_array = image.img_to_array(raw_img)
        raw_img = cv2.imread(CAPTCHA_DIR + name, flags=cv2.IMREAD_GRAYSCALE)
        img_array = raw_img.reshape((height, width, 1))
        print(img_array.shape)
        print(img_array)


if __name__ == '__main__':
    pic_deal()
