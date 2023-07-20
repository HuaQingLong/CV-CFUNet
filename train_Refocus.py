import tensorflow as tf
import numpy as np
# from matplotlib import pyplot as plt
import time
import sys
import datetime
import h5py

# import visualization
from CVmodels import encoder

image_height = 128
image_width = 128
image_channel = 1  # 复数为1 实数为2
# parameters
batch_size = 32
num_train_batches = 7500
print_step = 1500
num_prediction_steps = 20
LEARNING_RATES = [0.001, 0.0001, 0.00001]  # 学习率
BOUNDARIES = [3750, 6000]  # 学习衰减轮数

# num_train_batches = 200
# print_step = 50
# num_prediction_steps = 20
# LEARNING_RATES = [0.01, 0.001, 0.0001]  # 学习率
# BOUNDARIES = [100, 150]  # 学习衰减轮数

# 读取训练数据列表
train_anno_save_path = ".\\data\\voc_train.txt"
with open(train_anno_save_path, 'r') as f:
    train_anno = f.readlines()

train_anno = [x.replace('\n', '') for x in train_anno]
train_data_num = len(train_anno)

# 读取测试数据列表
test_anno_save_path = ".\\data\\voc_test.txt"
with open(test_anno_save_path, 'r') as f:
    test_anno = f.readlines()

test_anno = [x.replace('\n', '') for x in test_anno]
test_data_num = len(test_anno)

train_loss_log_path = ".\\data\\train_loss_log_512_3.txt"

def build_network(input, initial_state=None, initialize_to_zero=True):
    # 卷积

    images_r = tf.real(input)
    images_i = tf.imag(input)

    # 第一层的卷积层conv1，卷积核为3X3，有16个
    with tf.variable_scope('conv1') as scope:
        # 建立weights和biases的共享变量
        # conv1, shape = [kernel size, kernel size, channels, kernel numbers]
        weights_r = tf.get_variable('weights_r',
                                    shape=[4, 4, 1, 64],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        weights_i = tf.get_variable('weights_i',
                                    shape=[4, 4, 1, 64],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        biases_r = tf.get_variable('biases_r',
                                   shape=[64],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        biases_i = tf.get_variable('biases_i',
                                   shape=[64],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        conv_r = tf.nn.conv2d(images_r, weights_r, strides=[1, 2, 2, 1], padding='SAME') \
                 - tf.nn.conv2d(images_i, weights_i, strides=[1, 2, 2, 1], padding='SAME')
        conv_i = tf.nn.conv2d(images_r, weights_i, strides=[1, 2, 2, 1], padding='SAME') \
                 + tf.nn.conv2d(images_i, weights_r, strides=[1, 2, 2, 1], padding='SAME')

        pre_activation_r = tf.nn.bias_add(conv_r, biases_r)  # 加入偏差
        pre_activation_i = tf.nn.bias_add(conv_i, biases_i)  # 加入偏差

        conv1_r = tf.nn.tanh(pre_activation_r, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间
        conv1_i = tf.nn.tanh(pre_activation_i, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

        print('conv1:', conv1_r.shape)

    # 第二层的卷积层cov2，这里的命名空间和第一层不一样，所以可以和第一层取同名
    with tf.variable_scope('conv2') as scope:
        weights_r = tf.get_variable('weights_r',
                                    shape=[4, 4, 64, 128],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        weights_i = tf.get_variable('weights_i',
                                    shape=[4, 4, 64, 128],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        biases_r = tf.get_variable('biases_r',
                                   shape=[128],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        biases_i = tf.get_variable('biases_i',
                                   shape=[128],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        conv_r = tf.nn.conv2d(conv1_r, weights_r, strides=[1, 2, 2, 1], padding='SAME') \
                 - tf.nn.conv2d(conv1_i, weights_i, strides=[1, 2, 2, 1], padding='SAME')
        conv_i = tf.nn.conv2d(conv1_r, weights_i, strides=[1, 2, 2, 1], padding='SAME') \
                 + tf.nn.conv2d(conv1_i, weights_r, strides=[1, 2, 2, 1], padding='SAME')

        pre_activation_r = tf.nn.bias_add(conv_r, biases_r)  # 加入偏差
        pre_activation_i = tf.nn.bias_add(conv_i, biases_i)  # 加入偏差

        conv2_r = tf.nn.tanh(pre_activation_r, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间
        conv2_i = tf.nn.tanh(pre_activation_i, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

        print('conv2:', conv2_r.shape)

    # conv3
    with tf.variable_scope('conv3') as scope:
        weights_r = tf.get_variable('weights_r',
                                    shape=[4, 4, 128, 256],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        weights_i = tf.get_variable('weights_i',
                                    shape=[4, 4, 128, 256],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        biases_r = tf.get_variable('biases_r',
                                   shape=[256],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        biases_i = tf.get_variable('biases_i',
                                   shape=[256],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        conv_r = tf.nn.conv2d(conv2_r, weights_r, strides=[1, 2, 2, 1], padding='SAME') \
                 - tf.nn.conv2d(conv2_i, weights_i, strides=[1, 2, 2, 1], padding='SAME')
        conv_i = tf.nn.conv2d(conv2_r, weights_i, strides=[1, 2, 2, 1], padding='SAME') \
                 + tf.nn.conv2d(conv2_i, weights_r, strides=[1, 2, 2, 1], padding='SAME')

        pre_activation_r = tf.nn.bias_add(conv_r, biases_r)  # 加入偏差
        pre_activation_i = tf.nn.bias_add(conv_i, biases_i)  # 加入偏差

        conv3_r = tf.nn.tanh(pre_activation_r, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间
        conv3_i = tf.nn.tanh(pre_activation_i, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

        print('conv3:', conv3_r.shape)

    # conv4
    with tf.variable_scope('conv4') as scope:
        weights_r = tf.get_variable('weights_r',
                                    shape=[4, 4, 256, 512],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        weights_i = tf.get_variable('weights_i',
                                    shape=[4, 4, 256, 512],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        biases_r = tf.get_variable('biases_r',
                                   shape=[512],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        biases_i = tf.get_variable('biases_i',
                                   shape=[512],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        conv_r = tf.nn.conv2d(conv3_r, weights_r, strides=[1, 2, 2, 1], padding='SAME') \
                 - tf.nn.conv2d(conv3_i, weights_i, strides=[1, 2, 2, 1], padding='SAME')
        conv_i = tf.nn.conv2d(conv3_r, weights_i, strides=[1, 2, 2, 1], padding='SAME') \
                 + tf.nn.conv2d(conv3_i, weights_r, strides=[1, 2, 2, 1], padding='SAME')

        pre_activation_r = tf.nn.bias_add(conv_r, biases_r)  # 加入偏差
        pre_activation_i = tf.nn.bias_add(conv_i, biases_i)  # 加入偏差

        conv4_r = tf.nn.tanh(pre_activation_r, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间
        conv4_i = tf.nn.tanh(pre_activation_i, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

        print('conv4:', conv4_r.shape)


    # ConvGRU
    input_sequences_r = tf.expand_dims(conv4_r, axis=0)
    input_sequences_i = tf.expand_dims(conv4_i, axis=0)
    input_sequences = tf.complex(input_sequences_r, input_sequences_i)

    encoder_channels = [512, 512, 512]
    encoding_channels = encoder_channels[-1]
    with tf.variable_scope("encoder"):
        all_encoder_states, _ = encoder(inputs=input_sequences,
                                                          channels=encoder_channels,
                                                          initial_state=initial_state,
                                                          initialize_to_zero=initialize_to_zero)

    encoder_states = tf.squeeze(all_encoder_states, axis=0)
    encoder_states_r = tf.real(encoder_states)
    encoder_states_i = tf.imag(encoder_states)

    # 反卷积
    # conv4d
    with tf.variable_scope('conv4d') as scope:
        weights_r = tf.get_variable('weights_r',
                                    shape=[4, 4, 256, 512],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        weights_i = tf.get_variable('weights_i',
                                    shape=[4, 4, 256, 512],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        biases_r = tf.get_variable('biases_r',
                                   shape=[256],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        biases_i = tf.get_variable('biases_i',
                                   shape=[256],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        conv_r = tf.nn.conv2d_transpose(encoder_states_r, weights_r, output_shape=conv3_r.shape, strides=[1, 2, 2, 1],
                                        padding='SAME') \
                 - tf.nn.conv2d_transpose(encoder_states_i, weights_i, output_shape=conv3_r.shape, strides=[1, 2, 2, 1],
                                          padding='SAME')
        conv_i = tf.nn.conv2d_transpose(encoder_states_r, weights_i, output_shape=conv3_r.shape, strides=[1, 2, 2, 1],
                                        padding='SAME') \
                 + tf.nn.conv2d_transpose(encoder_states_i, weights_r, output_shape=conv3_r.shape, strides=[1, 2, 2, 1],
                                          padding='SAME')

        pre_activation_r = tf.nn.bias_add(conv_r, biases_r)  # 加入偏差
        pre_activation_i = tf.nn.bias_add(conv_i, biases_i)  # 加入偏差

        conv4d_r = tf.nn.tanh(pre_activation_r, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间
        conv4d_i = tf.nn.tanh(pre_activation_i, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

        conv4d_c_r = tf.concat([conv4d_r, conv3_r], axis=3)
        conv4d_c_i = tf.concat([conv4d_i, conv3_i], axis=3)

        print('conv4d:', conv4d_r.shape)

    # conv3d
    with tf.variable_scope('conv3d') as scope:
        weights_r = tf.get_variable('weights_r',
                                    shape=[4, 4, 128, 512],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        weights_i = tf.get_variable('weights_i',
                                    shape=[4, 4, 128, 512],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        biases_r = tf.get_variable('biases_r',
                                   shape=[128],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        biases_i = tf.get_variable('biases_i',
                                   shape=[128],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        conv_r = tf.nn.conv2d_transpose(conv4d_c_r, weights_r, output_shape=conv2_r.shape, strides=[1, 2, 2, 1],
                                        padding='SAME') \
                 - tf.nn.conv2d_transpose(conv4d_c_i, weights_i, output_shape=conv2_r.shape, strides=[1, 2, 2, 1],
                                          padding='SAME')
        conv_i = tf.nn.conv2d_transpose(conv4d_c_r, weights_i, output_shape=conv2_r.shape, strides=[1, 2, 2, 1],
                                        padding='SAME') \
                 + tf.nn.conv2d_transpose(conv4d_c_i, weights_r, output_shape=conv2_r.shape, strides=[1, 2, 2, 1],
                                          padding='SAME')

        pre_activation_r = tf.nn.bias_add(conv_r, biases_r)  # 加入偏差
        pre_activation_i = tf.nn.bias_add(conv_i, biases_i)  # 加入偏差

        conv3d_r = tf.nn.tanh(pre_activation_r, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间
        conv3d_i = tf.nn.tanh(pre_activation_i, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

        conv3d_c_r = tf.concat([conv3d_r, conv2_r], axis=3)
        conv3d_c_i = tf.concat([conv3d_i, conv2_i], axis=3)

        print('conv3d:', conv3d_r.shape)

    # conv2d
    with tf.variable_scope('conv2d') as scope:
        weights_r = tf.get_variable('weights_r',
                                    shape=[4, 4, 64, 256],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        weights_i = tf.get_variable('weights_i',
                                    shape=[4, 4, 64, 256],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        biases_r = tf.get_variable('biases_r',
                                   shape=[64],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        biases_i = tf.get_variable('biases_i',
                                   shape=[64],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        conv_r = tf.nn.conv2d_transpose(conv3d_c_r, weights_r, output_shape=conv1_r.shape, strides=[1, 2, 2, 1],
                                        padding='SAME') \
                 - tf.nn.conv2d_transpose(conv3d_c_i, weights_i, output_shape=conv1_r.shape, strides=[1, 2, 2, 1],
                                          padding='SAME')
        conv_i = tf.nn.conv2d_transpose(conv3d_c_r, weights_i, output_shape=conv1_r.shape, strides=[1, 2, 2, 1],
                                        padding='SAME') \
                 + tf.nn.conv2d_transpose(conv3d_c_i, weights_r, output_shape=conv1_r.shape, strides=[1, 2, 2, 1],
                                          padding='SAME')

        pre_activation_r = tf.nn.bias_add(conv_r, biases_r)  # 加入偏差
        pre_activation_i = tf.nn.bias_add(conv_i, biases_i)  # 加入偏差

        conv2d_r = tf.nn.tanh(pre_activation_r, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间
        conv2d_i = tf.nn.tanh(pre_activation_i, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

        conv2d_c_r = tf.concat([conv2d_r, conv1_r], axis=3)
        conv2d_c_i = tf.concat([conv2d_i, conv1_i], axis=3)

        print('conv2d:', conv2d_r.shape)

    # conv1d
    with tf.variable_scope('conv1d') as scope:
        weights_r = tf.get_variable('weights_r',
                                    shape=[4, 4, 1, 128],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        weights_i = tf.get_variable('weights_i',
                                    shape=[4, 4, 1, 128],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())  # stddev标准差

        biases_r = tf.get_variable('biases_r',
                                   shape=[1],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        biases_i = tf.get_variable('biases_i',
                                   shape=[1],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        conv_r = tf.nn.conv2d_transpose(conv2d_c_r, weights_r, output_shape=images_r.shape, strides=[1, 2, 2, 1],
                                        padding='SAME') \
                 - tf.nn.conv2d_transpose(conv2d_c_i, weights_i, output_shape=images_r.shape, strides=[1, 2, 2, 1],
                                          padding='SAME')
        conv_i = tf.nn.conv2d_transpose(conv2d_c_r, weights_i, output_shape=images_r.shape, strides=[1, 2, 2, 1],
                                        padding='SAME') \
                 + tf.nn.conv2d_transpose(conv2d_c_i, weights_r, output_shape=images_r.shape, strides=[1, 2, 2, 1],
                                          padding='SAME')

        conv1d_r = tf.nn.bias_add(conv_r, biases_r, name=scope.name)  # 加入偏差
        conv1d_i = tf.nn.bias_add(conv_i, biases_i, name=scope.name)  # 加入偏差

        # conv1d_r = tf.nn.relu(pre_activation_r, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间
        # conv1d_i = tf.nn.relu(pre_activation_i, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

        print('conv1d:', conv1d_r.shape)


    return tf.complex(conv1d_r, conv1d_i)


def run():
    # placeholders
    input_placeholder = tf.placeholder(dtype=tf.complex64,
                                     shape=(batch_size, image_height, image_width, image_channel))

    output_placeholder = tf.placeholder(dtype=tf.complex64,
                                      shape=(batch_size, image_height, image_width, image_channel))
    # output_sequences_rs = tf.reshape(output_sequences,
    #                                  shape=(-1, image_height, image_width, 1))

    global_step = tf.Variable(0, trainable=False)  # 初始化轮数计数器，定义为不可训练
    # 定义分段常数衰减的学习率（原论文：We start with a learning rate of 0.1, divide it by 10 at 32k and 48k iterations, and terminate training at 64k iterations）
    learning_rate = tf.train.piecewise_constant(global_step, boundaries=BOUNDARIES, values=LEARNING_RATES)
    # 定义训练过程
    # build network
    with tf.variable_scope("convlstm") as scope:
        predictions = build_network(input_placeholder)

        # loss and training
        with tf.variable_scope("trainer"):
            loss = tf.reduce_mean((tf.real(predictions) - tf.real(output_placeholder)) ** 2 + (tf.imag(predictions) - tf.imag(output_placeholder)) ** 2)
            # trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads, variables = zip(*optimizer.compute_gradients(loss))
            grads, global_norm = tf.clip_by_global_norm(grads, 0.1)
            trainer = optimizer.apply_gradients(zip(grads, variables))

        scope.reuse_variables()

    full_saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # construct summaries for tensorboard
    tf.summary.scalar('batch_loss', loss)
    summaries = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter('tensorboard/' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sess.graph)

    # get batch from data, assumes shape [time, batches, height, width]
    def generate_batch(i_batch):
        sardata = np.zeros((batch_size, image_height, image_width, image_channel)).astype(np.complex64)
        sarlabel = np.zeros((batch_size, image_height, image_width, image_channel)).astype(np.complex64)

        batch_inds = np.random.choice(train_data_num, batch_size, replace=False)
        # i_batch = i_batch % (int(train_data_num/batch_size))
        # batch_inds = np.array(range((i_batch*32), ((i_batch+1)*32)))

        for i, i_ind in zip(range(batch_size), batch_inds):
            i_train_anno = train_anno[i_ind]
            i_train_anno_list = i_train_anno.split()
            trainDataPath = i_train_anno_list[0]
            trainLabelPath = i_train_anno_list[1]

            trainData = h5py.File(trainDataPath)
            trainData = np.transpose(trainData['Im'])
            sardata[i, :, :, 0] = trainData['real'] + 1j * trainData['imag']

            trainLabel = h5py.File(trainLabelPath)
            trainLabel = np.transpose(trainLabel['Im'])
            sarlabel[i, :, :, 0] = trainLabel['real'] + 1j * trainLabel['imag']

        return sardata, sarlabel

    def generate_test_batch(test_batch_size, annoList):
        sardata = np.zeros((test_batch_size, image_height, image_width, image_channel)).astype(np.complex64)
        sarlabel = np.zeros((test_batch_size, image_height, image_width, image_channel)).astype(np.complex64)
        batch_inds = np.random.choice(test_data_num, test_batch_size, replace=False)
        for i, i_ind in zip(range(test_batch_size), batch_inds):
            i_train_anno = annoList[i_ind]
            i_train_anno_list = i_train_anno.split()
            trainDataPath = i_train_anno_list[0]
            trainLabelPath = i_train_anno_list[1]

            trainData = h5py.File(trainDataPath)
            trainData = np.transpose(trainData['Im'])
            sardata[i, :, :, 0] = trainData['real'] + 1j * trainData['imag']

            trainLabel = h5py.File(trainLabelPath)
            trainLabel = np.transpose(trainLabel['Im'])
            sarlabel[i, :, :, 0] = trainLabel['real'] + 1j * trainLabel['imag']
        return sardata, sarlabel

    # fig, axes = plt.subplots(1, 1)
    for i in range(num_train_batches):
        last_time = time.time()
        input_batch, output_batch = generate_batch(i)
        batch_feed = {input_placeholder: input_batch,
                      output_placeholder: output_batch}

        batch_summary, batch_loss, _, batch_predictions = \
            sess.run([summaries, loss, trainer, predictions],
                     feed_dict=batch_feed)

        # summary_writer.add_summary(batch_summary, i)
        sys.stdout.write("\rIteration: %i - loss %f - batches/s: %f" % (i, batch_loss, 1. / (time.time() - last_time)))
        sys.stdout.flush()

        with open(train_loss_log_path, 'a') as f:
            f.write(str(batch_loss) + "\n")

        if (i+1) % print_step == 0:
            test_batch_loss_total = 0
            for j in range(10):
                test_input_batch, test_output_batch = generate_test_batch(batch_size, test_anno)
                test_batch_feed = {input_placeholder: test_input_batch,
                                   output_placeholder: test_output_batch}

                test_batch_loss_total += sess.run(loss, feed_dict=test_batch_feed)
            test_batch_loss = test_batch_loss_total / 10

            ckpt_file = "./log_512_3/resnet_convgru_%i_train_loss=%.4f_test_loss=%.4f.ckpt" % (i, batch_loss, test_batch_loss)
            full_saver.save(sess, ckpt_file)


if __name__ == '__main__':
    run()
