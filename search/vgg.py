# -*- coding: utf-8 -*-
import keras
import numpy as np
from numpy import linalg as LA

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


class VGGNet:
    def __init__(self):
        keras.backend.clear_session()
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights = r'/home/ubuntu/jelly/opensrc/SearchImage/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        #加载自己训练的模型
        #self.model.load_weights(r'/home/ubuntu/jelly/opensrc/SearchImage/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)
        self.model.predict(np.zeros((1, 224, 224 , 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        #归一化
        norm_feat = feat[0]/LA.norm(feat[0])
        # print(feat.shape)
        return norm_feat

if __name__ == '__main__':
    url ='https://ss0.bdstatic.com/94oJfD_bAAcT8t7mm9GUKT-xh_/timg?image&quality=100&size=b4000_4000&sec=1606875555&di=1e4b4828776aa883d685f151370a571b&src=http://a4.att.hudong.com/27/67/01300000921826141299672233506.jpg'
    vgg=VGGNet()
    result =vgg.extract_feat(url)
    print(result)