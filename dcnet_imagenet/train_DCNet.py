from __future__ import print_function
import numpy as numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from loss import loss2

import alex2012_image_processing
from imagenet_data import ImagenetData

from DCNet_modified_ResNet import DCNet
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

FLAGS = tf.app.flags.FLAGS

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'imagenet-data', """XXXX""")
tf.app.flags.DEFINE_string('train_dir', 'log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 700001,
                            """Number of iterations to run.""")
tf.app.flags.DEFINE_string('model_file', 'model/DCNet_', """Directory to save model""")


is_training = tf.placeholder("bool")

train_set = ImagenetData(subset='train')
tr_images, tr_labels = alex2012_image_processing.distorted_inputs(train_set)

val_set  = ImagenetData(subset='validation')
val_images, val_labels = alex2012_image_processing.inputs(val_set)

images, labels = tf.cond(is_training, lambda: [tr_images, tr_labels], lambda: [val_images, val_labels])

cnn = VGG()
cnn.build(images, train_set.num_classes(), is_training)

fit_loss = loss2(cnn.score, labels, train_set.num_classes(), 'c_entropy') 
reg_loss = tf.add_n(tf.losses.get_regularization_losses())
orth_loss = tf.add_n(tf.get_collection('orth_constraint'))
loss_op = fit_loss + orth_loss + reg_loss

lr_ = tf.placeholder("float")

weight_list = [v for v in tf.trainable_variables() 
        if ('/filter' in v.name and 'score' not in v.name and 'shortcut' not in v.name)]
assign_op_list = []
for v in weight_list:
    assign_op_list.append(tf.assign(v, cnn.sphere_dict[v.name]))
assign_op = tf.group(*assign_op_list)

momentum = 0.9
update_op = tf.train.MomentumOptimizer(lr_, momentum).minimize(loss_op)

top1_op = tf.reduce_sum(tf.to_float(tf.nn.in_top_k(cnn.score, labels, 1)))
top5_op = tf.reduce_sum(tf.to_float(tf.nn.in_top_k(cnn.score, labels, 5)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

lr = 1e-1
for i in xrange(0, FLAGS.max_steps):

    if i==200000:
        lr = lr/10
    if i==375000:
        lr = lr/10
    if i==550000:
        lr = lr/10


    fit, reg, orth, top1, top5, _ = sess.run([fit_loss, reg_loss, orth_loss, top1_op, top5_op, update_op], 
                                        {lr_: lr, is_training: True})
    
    if i%500==0 and i!=0:
        print('====iteration_%d: fit=%.4f, reg=%.4f, orth=%.4f, top1=%.4f, top5=%.4f' 
            % (i, fit, reg, orth, top1/FLAGS.batch_size, top5/FLAGS.batch_size))

    if i % 100 == 0 and i != 0:
        sess.run(assign_op, {lr_: 0.0, is_training: False})

    if i%4000==0 and i!=0:
        n_test = val_set.num_examples_per_epoch()
        batch_size = FLAGS.batch_size
        top1= 0.0
        top5= 0.0
        for j in xrange(n_test/batch_size):
            a, b = sess.run([top1_op, top5_op], {is_training: False})
            top1 += a
            top5 += b
        top1 = top1/n_test
        top5 = top5/n_test
        print('++++iteration_%d: top1=%.4f, top5=%.4f' % (i, top1, top5))

    if i%100000==0 and i!=0:
        tf.train.Saver().save(sess, FLAGS.model_file+str(i))
