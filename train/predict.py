import os
import numpy as np
import cv2

from keras.models import load_model

from matplotlib import pyplot as plt

captcha_word = "0123456789abcdefghijklmnopqrstuvwxyz"

# 图片的长度和宽度
width = 160
height = 70

# 字符总数
word_class = len(captcha_word)

# 每个验证码所包含的字符数
word_len = 4

char_indices = dict((c, i) for i, c in enumerate(captcha_word))
indices_char = dict((i, c) for i, c in enumerate(captcha_word))


# 验证码字符串转数组
def captcha_to_vec(captcha):
    # 创建一个长度为 字符个数 * 字符种数 长度的数组
    vector = np.zeros(word_len * word_class)
    # 文字转成成数组
    for i, ch in enumerate(captcha):
        idex = i * word_class + char_indices[ch]
        vector[idex] = 1
    return vector


# 把数组转换回文字
def vec_to_captcha(vec):
    text = []
    # 把概率小于0.5的改为0，标记为错误
    vec[vec < 0.5] = 0

    char_pos = vec.nonzero()[0]

    for i, ch in enumerate(char_pos):
        text.append(captcha_word[ch % word_class])
    return ''.join(text)


def verify():
    CAPTCHA_PATH = 'E:\\pythonworkspace\\court\\server\\static\\uploads'
    model = load_model('output/captcha__model.hdf5')

    imgs = os.listdir(CAPTCHA_PATH)
    X_test = np.zeros((len(imgs), height, width, 1), dtype=np.float32)
    True_test = []
    for i, name in enumerate(imgs):
        img_path = os.path.join(CAPTCHA_PATH, name)
        img_array = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
        X_test[i] = img_array.reshape((height, width, 1))

        # result = model.predict(X_test)
        #
        # vex_test = vec_to_captcha(result[0])
        true_test = vec_to_captcha(captcha_to_vec(name.split('.')[0]))
        True_test.append(true_test)

        # plt.imshow(img_array)
        # plt.show()
        #
        # print('原始', true_test, '预测', vex_test)

    total_num = len(imgs)

    predict_right_num = 0
    predict_result = model.predict(X_test)
    for i, true_predict in enumerate(True_test):
        predict_result_name = vec_to_captcha(predict_result[i])
        print('原始', true_predict, '预测', predict_result_name)
        if true_predict == predict_result_name:
            predict_right_num += 1

    print('预测成功率：%.2f%%' % ((predict_right_num / total_num) * 100))


if __name__ == '__main__':
    verify()
