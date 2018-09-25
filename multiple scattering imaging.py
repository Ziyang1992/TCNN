#### This source code can be applied in the test in "Multiple-Scattering Media Imaging via End-to-End Neural Network"
#### Author: Ziyang Yuan and Hongxia Wang
#### Email: yuanziyang11@nudt.edu.cn
#### Date: 9.25.2018
#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import scipy.io as sio
import h5py
import matplotlib.pylab as plt
import copy
import datetime
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
iteration = 1
imagesize = 16
spectrumsize = 256
datanum = 3070
# The path where the data are stored
add1 = 'D:/TransmissionMatrices/TransmissionMatrices/Coherent_Data/AmpSLM_16x16/YH_squared_train.mat'
add2 = 'D:/TransmissionMatrices/TransmissionMatrices/Coherent_Data/AmpSLM_16x16/XH_train.mat'
add3 = 'D:/TransmissionMatrices/TransmissionMatrices/Coherent_Data/AmpSLM_16x16/YH_squared_test.mat'
add4 = 'D:/TransmissionMatrices/TransmissionMatrices/Coherent_Data/AmpSLM_16x16/XH_test.mat'
### To preprocess the data
x_add = h5py.File(add1)
y_add = h5py.File(add2)
z_add = h5py.File(add3)
h_add = h5py.File(add4)
x_train = x_add['YH_squared_train']
y_train = y_add['XH_train']
x_test = z_add['YH_squared_test']
y_test = h_add['XH_test']
x_train = np.transpose(x_train)
y_train = np.transpose(y_train)/255
x_test = np.sqrt(np.transpose(x_test))
y_test = np.transpose(y_test)/255
x_train = np.sqrt(x_train[0:datanum, :])
y_train = y_train[0:datanum, :]
training = True
trainable = True
batchsize = 1
type = 'test'
### chose the type "train" or "test"

### To define TCNN
if type == 'train':
    x = tf.placeholder(tf.float32, [None, spectrumsize*spectrumsize], name='spectrum')
    y_ = tf.placeholder(tf.float32, [None, imagesize*imagesize], name='image')
    keep_prob = tf.placeholder(tf.float32, name='keepprob')
    def weight_variables(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1, name=name)
        return tf.Variable(initial)

    def constant_variable(shape, name):
        return tf.Variable(tf.constant(0.0, shape=shape), name=name)

    def conv2d_1(x, W, name):
        initial = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)
        return initial

    def max_pool(x, size, pace, name):
        initial = tf.nn.max_pool(x, ksize=size, strides=pace, padding='SAME', name=name)
        return initial

    # The downsample block
    x_image = tf.reshape(x, [batchsize, spectrumsize, spectrumsize, 1],'F')
    with tf.name_scope('downsample') as scope:
        W_conv11 = weight_variables([3, 3, 1, 16], name='weight1')
        W_conv12 = weight_variables([3, 3, 1, 16], name='weight2')
        W_conv13 = weight_variables([3, 3, 1, 16], name='weight3')
        W_conv14 = weight_variables([3, 3, 1, 16], name='weight4')
        h_pool11 = conv2d_1(x_image, W_conv11, name='pool1')
        h_conv12 = conv2d_1(x_image, W_conv12, name='conv2')
        h_conv13 = conv2d_1(x_image, W_conv13, name='conv3')
        h_conv14 = conv2d_1(x_image, W_conv14, name='conv4')
        h_pool12 = max_pool(h_conv12,[1,2,2,1],[1,2,2,1], name='pool2')
        h_pool13 = max_pool(h_conv13,[1,4,4,1],[1,4,4,1], name='pool3')
        h_pool14 = max_pool(h_conv14,[1,8,8,1],[1,8,8,1], name='pool4')
    # The residue layer
    with tf.name_scope('residue') as scope:
        for i in range(0,4):
            W_conv21 = weight_variables([1, 1, 16, 16], name='weight11-'+str(i+1))
            W_conv22 = weight_variables([1, 1, 16, 16], name='weight21-'+str(i+1))
            W_conv23 = weight_variables([1, 1, 16, 16], name='weight31-'+str(i+1))
            W_conv24 = weight_variables([1, 1, 16, 16], name='weight41-'+str(i+1))
            h_conv21 = tf.nn.relu(
                tf.layers.batch_normalization(conv2d_1(h_pool11, W_conv21, name='None'), trainable=trainable, training=training), name='relu11-'+str(i+1))
            h_conv22 = tf.nn.relu(
                tf.layers.batch_normalization(conv2d_1(h_pool12, W_conv22, name='None'), trainable=trainable, training=training), name='relu21-'+str(i+1))
            h_conv23 = tf.nn.relu(
                tf.layers.batch_normalization(conv2d_1(h_pool13, W_conv23, name='None'), trainable=trainable, training=training), name='relu31-'+str(i+1))
            h_conv24 = tf.nn.relu(
                tf.layers.batch_normalization(conv2d_1(h_pool14, W_conv24, name='None'), trainable=trainable, training=training), name='relu41-'+str(i+1))
            h_pool111 = h_conv21
            h_pool112 = h_conv22
            h_pool113 = h_conv23
            h_pool114 = h_conv24
            W_conv21 = weight_variables([1, 1, 16, 16], name='weight12-'+str(i+1))
            W_conv22 = weight_variables([1, 1, 16, 16], name='weight22-'+str(i+1))
            W_conv23 = weight_variables([1, 1, 16, 16], name='weight32-'+str(i+1))
            W_conv24 = weight_variables([1, 1, 16, 16], name='weight42-'+str(i+1))
            h_conv21 = tf.nn.relu(tf.layers.batch_normalization(conv2d_1(h_pool111, W_conv21, name='None'), trainable=trainable, training=training), name='relu12-'+str(i+1))
            h_conv22 = tf.nn.relu(tf.layers.batch_normalization(conv2d_1(h_pool112, W_conv22, name='None'), trainable=trainable, training=training), name='relu22-'+str(i+1))
            h_conv23 = tf.nn.relu(tf.layers.batch_normalization(conv2d_1(h_pool113, W_conv23, name='None'), trainable=trainable, training=training), name='relu32-'+str(i+1))
            h_conv24 = tf.nn.relu(tf.layers.batch_normalization(conv2d_1(h_pool114, W_conv24, name='None'), trainable=trainable, training=training), name='relu42-'+str(i+1))
            h_pool11 = tf.add(h_conv21, h_pool11, name='residue1'+str(i+1))
            h_pool12 = tf.add(h_conv22, h_pool12, name='residue2'+str(i+1))
            h_pool13 = tf.add(h_conv23, h_pool13, name='residue3'+str(i+1))
            h_pool14 = tf.add(h_conv24, h_pool14, name='residue4'+str(i+1))

    # The upsample blocks
    with tf.name_scope('upsample') as scope:
        W_conv31 = weight_variables([3, 3, 16, 16], name='weight1')
        h_conv31 = conv2d_1(h_pool11, W_conv31, name='upsample1')
        h_upsample34 = h_pool14
        for i in range(1, 4):
            W_conv34 = weight_variables([3, 3, 16, 4*16], name='weight4-'+str(i+1))
            h_conv34 = tf.nn.relu(tf.layers.batch_normalization(conv2d_1(h_upsample34, W_conv34, name='None'), trainable=trainable, training=training), name='relu4-'+str(i+1))
            h_upsample34 = tf.depth_to_space(h_conv34, 2, name='upsample4-'+str(i+1))

        h_upsample33 = h_pool13
        for i in range(1, 3):
            W_conv33 = weight_variables([3, 3, 16, 4*16], name='weight3-'+str(i+1))
            h_conv33 = tf.nn.relu(tf.layers.batch_normalization(conv2d_1(h_upsample33, W_conv33, name='None'), trainable=trainable, training=training), name='relu3-'+str(i+1))
            h_upsample33 = tf.depth_to_space(h_conv33, 2, name='upsample3-'+str(i+1))
        h_upsample32 = h_pool12
        for i in range(1, 2):
            W_conv32 = weight_variables([3, 3, 16, 4*16], name='weight2-'+str(i+1))
            h_conv32 = tf.nn.relu(tf.layers.batch_normalization(conv2d_1(h_upsample32, W_conv32, name='None'), trainable=trainable, training=training), name='relu2-'+str(i+1))
            h_upsample32 = tf.depth_to_space(h_conv32, 2, name='upsample2-'+str(i+1))
    h_conv4 = tf.concat([h_conv31,h_upsample32, h_upsample33, h_upsample34], 3, name='combine')
    ### The transform block
    with tf.name_scope('transform') as scope:
        W_conv4 = weight_variables([1, 1, 16*4, 1], name='weight1')
        final = conv2d_1(h_conv4, W_conv4, name='conv1')
        refinal = tf.reshape(final, [batchsize, spectrumsize*spectrumsize], 'F')
        W1 = weight_variables([spectrumsize*spectrumsize, imagesize*imagesize], name='matrix')
        h_fc1_drop = tf.nn.dropout(refinal, keep_prob)
        refinal = tf.matmul(h_fc1_drop, W1, name='recover')
    ## build the loss function
    accuracy = tf.reduce_mean(tf.square(refinal-y_), name='loss')
    loss_summary = tf.summary.scalar("norm_square", accuracy)
    summary_op = tf.summary.merge_all()
    ## start up the session
    sess = tf.InteractiveSession(config=config)
    print("Setting up Saver...")
    saver = tf.train.Saver()
    ### The path to store the ckpt files
    ckpt = tf.train.get_checkpoint_state('D:/github/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    ### The path to store the tensorboard files
    train_writer = tf.summary.FileWriter('D:/github/train', sess.graph)
    validation_writer = tf.summary.FileWriter('D:/github/validation')
    kfolds = np.int(x_train.shape[0]/10)
    x_train1 = x_train[0:(x_train.shape[0]-kfolds)]
    y_train1 = y_train[0:(x_train.shape[0]-kfolds)]
    x_validation1 = x_train[(x_train.shape[0]-kfolds):x_train.shape[0]]
    y_validation1 = y_train[(x_train.shape[0]-kfolds):x_train.shape[0]]
    batchnum = int(x_train1.shape[0] / batchsize)
    global_step = tf.Variable(1)
    learningrate = tf.train.exponential_decay(1e-3, global_step, batchnum, 0.98, staircase=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
       train_step = tf.train.AdamOptimizer(learningrate).minimize(accuracy, global_step=global_step)
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    time = 0
    plt.ion()
    plt.figure()
    loop = 0
    for i in range(iteration):
        times = i % batchnum
        batch1 = x_train1[0+times*batchsize:batchsize+times*batchsize, :]
        batch2 = y_train1[0+times*batchsize:batchsize+times*batchsize, :]
        if i % 10 == 0:
            train_accuracy, summary_str = sess.run([accuracy, loss_summary], feed_dict={x: batch1, y_: batch2, keep_prob : 0.5})
            train_writer.add_summary(summary_str, i)
            print("step %d, training accuracy%g"%(i,train_accuracy))
        if i % 100 == 0:
            training = False
            trainable = False
            indice1 = np.random.permutation(kfolds-batchsize)
            vbatch1 = x_validation1[indice1[0] % kfolds:(batchsize + indice1[0]) % kfolds, :]
            vbatch2 = y_validation1[indice1[0] % kfolds:(batchsize + indice1[0]) % kfolds, :]
            test_accuracy, summary_str1 = sess.run([accuracy, loss_summary], feed_dict={x: vbatch1, y_: vbatch2,keep_prob : 1})
            print("step %d, test accuracy%g" % (i, test_accuracy))
            validation_writer.add_summary(summary_str1, i)
            c = refinal.eval(feed_dict={x: x_test[time % y_test.shape[0]:time % y_test.shape[0]+1, :], y_: y_test[time%y_test.shape[0]:time%y_test.shape[0]+1, :], keep_prob : 1})
            plt.imshow(np.reshape(c[0, :], [imagesize, imagesize], 'F'), cmap='Greys_r')
            time = time + 1
            plt.show()
            plt.pause(1)
            plt.close()
            training = True
            trainable = True
        train_step.run(feed_dict={x: batch1, y_: batch2, keep_prob : 0.5})
        if times == (batchnum-1):
            indice = np.random.permutation(x_train1.shape[0])
            x_trainn = copy.deepcopy(x_train1[indice])
            y_trainn = copy.deepcopy(y_train1[indice])
            x_train1 = copy.deepcopy(x_trainn)
            y_train1 = copy.deepcopy(y_trainn)
            path = 'D:/github/model.ckpt'
            save_path = saver.save(sess, path, loop)
            loop = loop + 1
        global_step = i
    path = 'D:/github/model.ckpt'
    save_path = saver.save(sess, path)
if type == 'test':
    logs_dir = 'D:/github'
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    print("Setting up Saver...")
    sess = tf.Session()
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    graph = tf.get_default_graph()
    spectrum = graph.get_tensor_by_name('spectrum:0')
    image = graph.get_tensor_by_name('image:0')
    keep_prob = graph.get_tensor_by_name('keepprob:0')
    w1 = graph.get_tensor_by_name('residue/weight11-1:0')
    w2 = graph.get_tensor_by_name('residue/weight11-2:0')
    recover1 = graph.get_tensor_by_name('transform/recover:0')
    start = datetime.datetime.now()
    testnum = x_test.shape[0]
    recover2 = np.zeros([testnum, imagesize*imagesize])
    for time in range(testnum):
        recover2[time, :] = sess.run(recover1, feed_dict={spectrum: x_test[time%testnum:time%testnum+1, :], image: y_test[time%testnum:time%testnum+1, :], keep_prob:1})
    end = datetime.datetime.now()
    print("%s --->" % (end-start))
