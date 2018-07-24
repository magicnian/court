#!/usr/bin/env python
# -*- coding: utf-8 -*-

from server import predict_util, constant
from flask import Flask, request, jsonify
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
import os, base64
import uuid
import logging

logger = None

height = 70
width = 160

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
model = None
graph = None


@app.errorhandler(500)
def handle_500():
    logger.error('系统异常')
    return jsonify({'code': constant.FAILED_CODE, 'msg': '系统错误！', 'result': ''})


def cache_model():
    global model
    model = load_model('model/captcha__model.hdf5')
    global graph
    graph = tf.get_default_graph()


def prepare_img(img_path):
    im = cv2.imread(img_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
    return thresh


def verify(img_path):
    with graph.as_default():
        X_test = np.zeros((1, height, width, 1), dtype=np.float32)
        img_array = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
        X_test[0] = img_array.reshape((height, width, 1))

        y_test = model.predict(X_test)

        y_test_name = predict_util.vec_to_captcha(y_test[0])

        return y_test_name


@app.route("/predict", methods=["POST"])
def predict():
    img_type = request.form['type']
    if not img_type:
        return jsonify({'code': constant.FAILED_CODE, 'msg': '缺少type参数', 'result': ''})

    '''失信和失信被执行人'''
    if img_type == 'zznj_shixin':
        base64_str = request.form['img']
        if base64_str:
            '''保存图片至本地'''
            img_data = base64.b64decode(base64_str)
            img_name = uuid.uuid1()
            img_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], str(img_name) + '.png')
            f = open(img_path, 'wb')
            f.write(img_data)
            f.close()
            thresh = prepare_img(img_path)
            cv2.imwrite(img_path, thresh)
            result = verify(img_path)
            logger.info('predict result: ' + result)
            os.remove(img_path)
            return jsonify({'code': constant.SUCCESS_CODE, 'msg': '解析成功', 'result': result})
        else:
            return jsonify({'code': constant.FAILED_CODE, 'msg': '图片数据为空', 'result': ''})
    else:
        return jsonify({'code': constant.FAILED_CODE, 'msg': '不支持的type类型', 'result': ''})


if __name__ == '__main__':
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename='log/server.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    logger.info("* Loading Keras model and Flask starting server..."
                "please wait until server has fully started")

    cache_model()
    app.run(host='0.0.0.0', port=8333)
