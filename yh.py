import cv2
import tensorflow as tf

#
# img = cv2.imread('/media/yang/F/DataSet/Face/LFW_FF-GAN/LFW_FF-GAN/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')
# cv2.imshow('ff',img)
# image = cv2.resize(img,(227,227),interpolation=cv2.INTER_CUBIC)
# cv2.imshow('ff1',image)

img = tf.get_variable('dada',initializer=tf.random_normal(shape=[1,2,2,3],dtype=tf.float32), dtype=tf.float32)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('o:',sess.run(img))

    R = tf.reshape(img[:,:,:,0],[1,2,2,1])
    G = tf.reshape(img[:,:,:,1],[1,2,2,1])
    B = tf.reshape(img[:, :, :, 2], [1, 2, 2, 1])
    img = tf.concat([B,G,R],axis=3)
    pr = sess.run(img)
    print(pr)

x = tf.image.resize_images(x, [227, 227])

# x is RGB and is value range is [-1,1].
# first we need to change RGB to BGR;
batch_, height_, width_, nc = x.get_shape().as_list()
R = tf.reshape(x[:, :, :, 0], [batch_, height_, width_, 1])
G = tf.reshape(x[:, :, :, 1], [batch_, height_, width_, 1])
B = tf.reshape(x[:, :, :, 2], [batch_, height_, width_, 1])
x = tf.concat([B, G, R], axis=3)
