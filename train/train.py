import numpy as np
import os
import cv2
import tensorflow as tf

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

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


test_vec = captcha_to_vec("2we4")
vec_test = vec_to_captcha(test_vec)

print(test_vec)
print(vec_test)

train_dir = 'H:\\document\\ocr\\dst'

image_list = []

for item in os.listdir(train_dir):
    image_list.append(item)

np.random.shuffle(image_list)

X = np.zeros((len(image_list), height, width, 1), dtype=np.uint8)

y = np.zeros((len(image_list), word_len * word_class), dtype=np.uint8)

for i, img in enumerate(image_list):
    if i % 100 == 0:
        print(i)
    img_path = os.path.join(train_dir, img)
    img_array = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    X[i] = img_array.reshape((height, width, 1))
    result = img.split('-')[0]
    y[i] = captcha_to_vec(result)

# 创建输入，结构为 高，宽，通道
input_tensor = Input(shape=(height, width, 1))

x = input_tensor

# 构建卷积网络
# 两层卷积层，一层池化层，重复3次。因为生成的验证码比较小，padding使用same
x = Convolution2D(32, 3, padding='same', activation='relu')(x)
x = Convolution2D(32, 3, padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Convolution2D(64, 3, padding='same', activation='relu')(x)
x = Convolution2D(64, 3, padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Convolution2D(128, 3, padding='same', activation='relu')(x)
x = Convolution2D(128, 3, padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
x = Flatten()(x)

# 为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，Dropout层用于防止过拟合。
x = Dropout(0.25)(x)

# Dense就是常用的全连接层
x = [Dense(len(captcha_word), activation='softmax', name='c%d' % (i + 1))(x) for i in range(word_len)]

output = concatenate(x)

# 构建模型
model = Model(inputs=input_tensor, outputs=output)
# 这里优化器选用Adadelta，学习率0.1
opt = Adadelta(lr=0.1)

# 编译模型以供训练，损失函数使用 categorical_crossentropy，使用accuracy评估模型在训练和测试时的性能的指标
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 每次epoch都保存一下权重，用于继续训练
checkpointer = ModelCheckpoint(filepath="output/model.{epoch:02d}--{val_loss:.2f}-{val_acc:.4f}.hdf5",
                               verbose=2, save_weights_only=False)

model.fit(X, y, batch_size=128, epochs=100, callbacks=[checkpointer], validation_split=0.01)

# 保存权重和模型
model.save_weights('output/captcha_model_weights.hdf5')
model.save('output/captcha__model.hdf5')
