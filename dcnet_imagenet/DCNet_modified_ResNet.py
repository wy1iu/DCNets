from __future__ import print_function
import tensorflow as tf
import numpy as numpy

class DCNet():
    def get_conv_filter(self, shape, reg, stddev, xavier=False):
        if xavier:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = tf.random_normal_initializer(stddev=stddev)
        if reg:
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            filt = tf.get_variable('filter', shape, initializer=init, regularizer=regu)
        else:
            filt = tf.get_variable('filter', shape, initializer=init)

        return filt      

    def get_bias(self, dim, init_bias, name):
        with tf.variable_scope(name):
            init = tf.constant_initializer(init_bias)
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            bias = tf.get_variable('bias', dim, initializer=init, regularizer=regu)

            return bias

    def xnorm_bn(self, x, phase_train):
        with tf.variable_scope('bn'):

            batch_mean = tf.reduce_mean(x, axis=[0,1,2], keep_dims=True)
            ema = tf.train.ExponentialMovingAverage(decay=0.999)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean)

            mean = tf.cond(phase_train,
                           mean_var_with_update,
                           lambda: (ema.average(batch_mean)))
            return mean + 1e-6

    def batch_norm(self, x, n_out, phase_train):
        with tf.variable_scope('bn'):

            gamma = self.get_bias(n_out, 1.0, 'gamma')
            beta = self.get_bias(n_out, 0.0, 'beta')

            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.999)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    def _max_pool(self, bottom, ksize, name):
        return tf.nn.max_pool(bottom, ksize=[1, ksize, ksize, 1], strides=[1, 2, 2, 1],
            padding='SAME', name=name)

    def _get_filter_norm(self, filt):
        eps = 1e-4
        return tf.sqrt(tf.reduce_sum(filt*filt, [0, 1, 2], keep_dims=True)+eps)

    def _get_input_norm(self, bottom, ksize, stride, pad):
        eps = 1e-4
        shape = [ksize, ksize, bottom.get_shape()[3], 1]
        filt = tf.ones(shape)
        input_norm = tf.sqrt(tf.nn.conv2d(bottom*bottom, filt, [1,stride,stride,1], padding=pad)+eps)
        return input_norm    

    def _add_orthogonal_constraint(self, filt, n_filt):
        filt = tf.reshape(filt, [-1, n_filt])
        inner_pro = tf.matmul(tf.transpose(filt), filt)
        loss = 1e-5*tf.nn.l2_loss(inner_pro-tf.eye(n_filt))
        tf.add_to_collection('orth_constraint', loss)

    def prelu(self,_x):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - tf.abs(_x)) * 0.5
        return pos + neg

    def _conv_layer(self, bottom, ksize, n_filt, is_training, name, stride=1, bn=True, relu=True, pad='SAME', 
                     norm=True, reg=False, orth=False, w_norm=False, xnorm_bn=True, init_gau=True):
        with tf.variable_scope(name) as scope:
            n_input = bottom.get_shape().as_list()[3]
            shape = [ksize, ksize, n_input, n_filt]
            print("shape of filter %s: %s" % (name, str(shape)))
            if init_gau:
                filt = self.get_conv_filter(shape, reg, stddev=0.01, xavier=False)
            else:
                filt = self.get_conv_filter(shape, reg, stddev=0.01, xavier=True)
            if norm:
                wnorm = self._get_filter_norm(filt)
                filt_name = filt.name
                filt /= wnorm
                self.sphere_dict[filt_name] = filt
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding=pad)

            if w_norm:
                conv = conv/self._get_filter_norm(filt)
            if norm:
                xnorm = self._get_input_norm(bottom, ksize, stride, pad)
                conv /= xnorm
                radius = tf.get_variable('radius', shape=(1,1,1,n_filt), initializer=tf.constant_initializer(1.0))**2 + 1e-4
                if xnorm_bn:
                    xnorm_mean = self.xnorm_bn(xnorm, is_training)
                    conv *= tf.tanh(xnorm / xnorm_mean / radius)
                else:
                    conv *= tf.tanh(xnorm / radius)
            if orth:
                self._add_orthogonal_constraint(filt, n_filt)
            if bn:
                conv = self.batch_norm(conv, n_filt, is_training)
            if relu:
                return self.prelu(conv)
            else:
                return conv

    def _resnet_unit_v1(self, bottom, ksize, n_filt, is_training, name, stride, norm, reg, orth=False, w_norm=False, xnorm_bn=True):
        with tf.variable_scope(name):

            residual = self._conv_layer(bottom, ksize, n_filt, is_training, 'first', 
                                        stride=1, bn=True, relu=True, norm=norm, reg=reg, orth=orth, w_norm=w_norm, xnorm_bn=xnorm_bn, init_gau=True)
            residual = self._conv_layer(residual, ksize, n_filt, is_training, name='second', 
                                        stride=1, bn=True, relu=False, norm=norm, reg=reg, orth=orth, w_norm=w_norm, xnorm_bn=xnorm_bn, init_gau=True)
            shortcut = bottom
            return residual + shortcut

    # Input should be an rgb image [batch, height, width, 3]
    def build(self, rgb, n_class, is_training):        
        self.wd = 1e-4
        self.sphere_dict = {}

        # 224X224
        ksize = 7
        n_out = 128
        feat = self._conv_layer(rgb, ksize, n_out, is_training, name='root', stride=2, bn=True, relu=True, pad='SAME', norm=True, reg=False, orth=True, w_norm=False, xnorm_bn=True, init_gau=False)

        # 112X112
        ksize = 3
        feat = self._max_pool(feat, ksize, 'max_pooling')

        # 56X56
        n_out = 128
        n_unit= 1
        for i in range(n_unit):
            feat = self._resnet_unit_v1(feat, ksize, n_out, is_training, name='block1_unit'+str(i), stride=1, norm=True, reg=False, orth=True, w_norm=False, xnorm_bn=True)

        n_out = 256
        feat = self._conv_layer(feat, ksize, n_out, is_training, name='conv2', stride=2, bn=True, relu=True, pad='SAME', norm=True, reg=False, orth=True, w_norm=False, xnorm_bn=True, init_gau=False)
        n_unit= 2
        # 28X28
        for i in range(n_unit):
            feat = self._resnet_unit_v1(feat, ksize, n_out, is_training, name='block2_unit'+str(i), stride=1, norm=True, reg=False, orth=True, w_norm=False, xnorm_bn=True)

        n_out = 512
        n_unit= 3
        feat = self._conv_layer(feat, ksize, n_out, is_training, name='conv3', stride=2, bn=True, relu=True, pad='SAME', norm=True, reg=False, orth=True, w_norm=False, xnorm_bn=True, init_gau=False)
        # 14X14
        for i in range(n_unit):
            feat = self._resnet_unit_v1(feat, ksize, n_out, is_training, name='block3_unit'+str(i), stride=1, norm=True, reg=False, orth=True, w_norm=False, xnorm_bn=True)
        
        n_out = 1024
        n_unit= 1
        feat = self._conv_layer(feat, ksize, n_out, is_training, name='conv4', stride=2, bn=True, relu=True, pad='SAME', norm=True, reg=False, orth=True, w_norm=False, xnorm_bn=True, init_gau=False)
        # 7X7
        # invalid in modified ResNet-18
        for i in range(n_unit):
            feat = self._resnet_unit_v1(feat, ksize, n_out, is_training, name='block4_unit'+str(i), stride=1, norm=True, reg=False, orth=True, w_norm=False, xnorm_bn=True)

        feat = tf.nn.avg_pool(feat, [1,7,7,1], [1,1,1,1], 'VALID')

        self.score = tf.squeeze(self._conv_layer(feat, 1, n_class, is_training, "score", bn=False, relu=False, pad='VALID', norm=False, reg=True, orth=False, w_norm=False, xnorm_bn=False, init_gau=False))
        self.pred = tf.argmax(self.score, axis=1)
