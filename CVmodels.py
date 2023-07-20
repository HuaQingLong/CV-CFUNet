import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from CVConvGRUCell import BasicConvLSTM

def encoder(inputs, channels, initial_state=None, initialize_to_zero=True):
    batch_size = tf.shape(inputs)[0]
    input_shape = inputs.shape.as_list()[2:]

    encoder_lstm_cells = [BasicConvLSTM(conv_ndims=2,
                                        input_shape=input_shape,
                                        kernel_shape=[3, 3],
                                        output_channels=num_channels) for num_channels in channels]

    encoder_lstm = tf.contrib.rnn.MultiRNNCell(encoder_lstm_cells)

    if initialize_to_zero:
        initial_state_LSTM_r = encoder_lstm.zero_state(batch_size, tf.float32)
        initial_state_LSTM_i = encoder_lstm.zero_state(batch_size, tf.float32)
        initial_state = []
        for i in range(len(initial_state_LSTM_r)):
            _, hidden_r = initial_state_LSTM_r[i]
            _, hidden_i = initial_state_LSTM_i[i]
            hidden = tf.complex(hidden_r, hidden_i)
            initial_state.append(hidden)

        initial_state = tuple(initial_state)

    all_encoder_states, final_encoder_state = tf.nn.dynamic_rnn(cell=encoder_lstm,
                                                                inputs=inputs,
                                                                initial_state=initial_state,
                                                                dtype=tf.complex64)
    return all_encoder_states, final_encoder_state

# from CVConvLSTMCell import BasicConvLSTM
#
# def encoder(inputs, channels, initial_state=None, initialize_to_zero=True):
#     batch_size = tf.shape(inputs)[0]
#     input_shape = inputs.shape.as_list()[2:]
#
#     encoder_lstm_cells = [BasicConvLSTM(conv_ndims=2,
#                                         input_shape=input_shape,
#                                         kernel_shape=[3, 3],
#                                         output_channels=num_channels) for num_channels in channels]
#
#     encoder_lstm = tf.contrib.rnn.MultiRNNCell(encoder_lstm_cells)
#
#     if initialize_to_zero:
#         initial_state_r = encoder_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
#         initial_state_i = encoder_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
#         initial_state = []
#         for i in range(len(initial_state_r)):
#             cell_r, hidden_r = initial_state_r[i]
#             cell_i, hidden_i = initial_state_i[i]
#             cell2 = tf.complex(cell_r, cell_i)
#             hidden = tf.complex(hidden_r, hidden_i)
#             initial_state.append(rnn_cell_impl.LSTMStateTuple(cell2, hidden))
#
#         initial_state = tuple(initial_state)
#
#     all_encoder_states, final_encoder_state = tf.nn.dynamic_rnn(cell=encoder_lstm,
#                                                                 inputs=inputs,
#                                                                 initial_state=initial_state,
#                                                                 dtype=tf.complex64)
#     return all_encoder_states, final_encoder_state



def output_layers(decoder_state):
    inputChannels = decoder_state.shape[-1]
    input_r = tf.real(decoder_state)
    input_i = tf.imag(decoder_state)

    with tf.variable_scope('conv1') as scope:
        weights_r = tf.get_variable('weights_r',
                                    shape=[3, 3, inputChannels, 16],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        weights_i = tf.get_variable('weights_i',
                                    shape=[3, 3, inputChannels, 16],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        biases_r = tf.get_variable('biases_r',
                                   shape=[16],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        biases_i = tf.get_variable('biases_i',
                                   shape=[16],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        conv_r = tf.nn.conv2d(input_r, weights_r, strides=[1, 1, 1, 1], padding='SAME') \
                 - tf.nn.conv2d(input_i, weights_i, strides=[1, 1, 1, 1], padding='SAME')

        conv_i = tf.nn.conv2d(input_r, weights_i, strides=[1, 1, 1, 1], padding='SAME') \
                 + tf.nn.conv2d(input_i, weights_r, strides=[1, 1, 1, 1], padding='SAME')

        pre_activation_r = tf.nn.bias_add(conv_r, biases_r)  # 加入偏差
        pre_activation_i = tf.nn.bias_add(conv_i, biases_i)  # 加入偏差

        conv1_r = tf.nn.relu(pre_activation_r, name='conv1')
        conv1_i = tf.nn.relu(pre_activation_i, name='conv1')

    with tf.variable_scope('conv2') as scope:
        weights_r = tf.get_variable('weights_r',
                                    shape=[3, 3, 16, 8],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        weights_i = tf.get_variable('weights_i',
                                    shape=[3, 3, 16, 8],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        biases_r = tf.get_variable('biases_r',
                                   shape=[8],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        biases_i = tf.get_variable('biases_i',
                                   shape=[8],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        conv_r = tf.nn.conv2d(conv1_r, weights_r, strides=[1, 1, 1, 1], padding='SAME') \
                 - tf.nn.conv2d(conv1_i, weights_i, strides=[1, 1, 1, 1], padding='SAME')

        conv_i = tf.nn.conv2d(conv1_r, weights_i, strides=[1, 1, 1, 1], padding='SAME') \
                 + tf.nn.conv2d(conv1_i, weights_r, strides=[1, 1, 1, 1], padding='SAME')

        pre_activation_r = tf.nn.bias_add(conv_r, biases_r)  # 加入偏差
        pre_activation_i = tf.nn.bias_add(conv_i, biases_i)  # 加入偏差

        conv2_r = tf.nn.relu(pre_activation_r, name='conv2')
        conv2_i = tf.nn.relu(pre_activation_i, name='conv2')

    with tf.variable_scope('conv3') as scope:
        weights_r = tf.get_variable('weights_r',
                                    shape=[1, 1, 8, 1],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        weights_i = tf.get_variable('weights_i',
                                    shape=[1, 1, 8, 1],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        biases_r = tf.get_variable('biases_r',
                                   shape=[1],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        biases_i = tf.get_variable('biases_i',
                                   shape=[1],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

        conv_r = tf.nn.conv2d(conv2_r, weights_r, strides=[1, 1, 1, 1], padding='SAME') \
                 - tf.nn.conv2d(conv2_i, weights_i, strides=[1, 1, 1, 1], padding='SAME')

        conv_i = tf.nn.conv2d(conv2_r, weights_i, strides=[1, 1, 1, 1], padding='SAME') \
                 + tf.nn.conv2d(conv2_i, weights_r, strides=[1, 1, 1, 1], padding='SAME')

        conv3_r = tf.nn.bias_add(conv_r, biases_r)  # 加入偏差
        conv3_i = tf.nn.bias_add(conv_i, biases_i)  # 加入偏差

        # conv3_r = tf.nn.relu(pre_activation_r, name='conv3')
        # conv3_i = tf.nn.relu(pre_activation_i, name='conv3')

        # predictions_flat = tf.contrib.slim.conv2d(decoder_state, 16, 3, padding='SAME', activation_fn=tf.nn.relu)
        # predictions_flat = tf.contrib.slim.conv2d(predictions_flat, 8, 3, padding='SAME', activation_fn=tf.nn.relu)
        # predictions_flat = tf.contrib.slim.conv2d(predictions_flat, 1, 1, padding='SAME', activation_fn=None)

    predictions_flat = tf.complex(conv3_r, conv3_i)

    return predictions_flat

# 定义卷积函数
def myconv3d(x_r, x_i, shape, strides=1, padding="SAME"):
    w_r = tf.get_variable("w_r", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))  # 生成去掉最大偏离点的正态分布的随机数
    w_i = tf.get_variable("w_i", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))  # 生成去掉最大偏离点的正态分布的随机数

    conv_r = tf.nn.conv3d(x_r, w_r, strides=[1, 1, strides, strides, 1], padding=padding) \
             - tf.nn.conv3d(x_i, w_i, strides=[1, 1, strides, strides, 1], padding=padding)

    conv_i = tf.nn.conv3d(x_r, w_i, strides=[1, 1, strides, strides, 1], padding=padding) \
             + tf.nn.conv3d(x_i, w_r, strides=[1, 1, strides, strides, 1], padding=padding)

    return conv_r, conv_i


# 定义批规范化函数
# 训练时使用tf.nn.moments函数来计算批数据的均值和方差，然后在迭代过程中更新均值和方差的分布，并且使用tf.nn.batch_normalization做标准化
# 使用with tf.control_dependencies...语句结构块来强迫Tensorflow先更新均值和方差的分布，再执行批标准化操作
# 测试时使用的均值和方差分布来自于训练时使用滑动平均算法估计的值
def batch_normalization(x, depth, is_training, dataType):
    gamma = tf.get_variable("gamma"+dataType, [depth], initializer=tf.ones_initializer)
    beta = tf.get_variable("beta"+dataType, [depth], initializer=tf.zeros_initializer)
    pop_mean = tf.get_variable("mean"+dataType, [depth], initializer=tf.zeros_initializer, trainable=False)
    pop_variance = tf.get_variable("variance"+dataType, [depth], initializer=tf.ones_initializer, trainable=False)
    if is_training:
        batch_mean, batch_variance = tf.nn.moments(x, [0, 1, 2, 3], keep_dims=False)
        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))
        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(x, batch_mean, batch_variance, beta, gamma, 1e-3)
    else:
        return tf.nn.batch_normalization(x, pop_mean, pop_variance, beta, gamma, 1e-3)


# 定义残差模块
def basicblock(x_r, x_i, depth, image_frames, kernel_size, kernel_num, is_training, strides=1, residual_path=False):
    residual_r = x_r  # residual等于输入值本身，即residual=x
    residual_i = x_i  # residual等于输入值本身，即residual=x
    # 将输入通过卷积、BN层、激活层，计算F(x)
    with tf.variable_scope('1'):
        c1_r, c1_i = myconv3d(x_r, x_i, [image_frames, kernel_size, kernel_size, depth, kernel_num], strides)
        b1_r = batch_normalization(c1_r, kernel_num, is_training, "_r")
        b1_i = batch_normalization(c1_i, kernel_num, is_training, "_i")
        a1_r = tf.nn.relu(b1_r)
        a1_i = tf.nn.relu(b1_i)

    with tf.variable_scope('2'):
        c2_r, c2_i = myconv3d(a1_r, a1_i, [image_frames, kernel_size, kernel_size, kernel_num, kernel_num])
        b2_r = batch_normalization(c2_r, kernel_num, is_training, "_r")
        b2_i = batch_normalization(c2_i, kernel_num, is_training, "_i")

    # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
    if residual_path:
        with tf.variable_scope('3'):
            down_c1_r, down_c1_i = myconv3d(x_r, x_i, [image_frames, 1, 1, depth, kernel_num], strides)
            residual_r = batch_normalization(down_c1_r, kernel_num, is_training, "_r")
            residual_i = batch_normalization(down_c1_i, kernel_num, is_training, "_i")

    a2_r = tf.nn.relu(b2_r + residual_r)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
    a2_i = tf.nn.relu(b2_i + residual_i)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
    return a2_r, a2_i


# 定义全连接函数
def fc(x_r, x_i, shape):
    w_r = tf.get_variable("w_r", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))  # 生成去掉最大偏离点的正态分布的随机数
    w_i = tf.get_variable("w_i", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))  # 生成去掉最大偏离点的正态分布的随机数
    b_r = tf.get_variable("b_r", shape[1], initializer=tf.constant_initializer(0.0))
    b_i = tf.get_variable("b_i", shape[1], initializer=tf.constant_initializer(0.0))

    local_r = tf.matmul(x_r, w_r) - tf.matmul(x_i, w_i)
    local_i = tf.matmul(x_r, w_i) + tf.matmul(x_i, w_r)

    pre_activation_r = tf.nn.bias_add(local_r, b_r)  # 加入偏差
    pre_activation_i = tf.nn.bias_add(local_i, b_i)  # 加入偏差

    return pre_activation_r, pre_activation_i


# 定义前向传播函数
# 初始化各层的权重，设定计算式（计算图只规定计算式，不进行真实计算）
def resnet_forward(x, IMAGE_FRAMES, KERNEL_SIZE, IMAGE_CHANNELS, is_training=True):

    # IMAGE_CHANNELS = 2  # 输入图片的深度
    # KERNEL_SIZE = 3  # 卷积核的大小（所有卷积核的大小都一样）
    # IMAGE_FRAMES = 10  # 输入图片的帧数
    # KERNEL_NUM = [64, 128, 256, 512]  # 残差模块各个block卷积核的个数
    # DEPTH = [[64, 64], [64, 128], [128, 256], [256, 512]]  # 残差模块各个block输入图片的深度
    KERNEL_NUM = [16, 32, 64, 128]  # 残差模块各个block卷积核的个数
    DEPTH = [[16, 16], [16, 32], [32, 64], [64, 128]]  # 残差模块各个block输入图片的深度
    BLOCK_LIST = [2, 2, 2, 2]  # 残差模块各个block卷积层的数量

    x_r = tf.real(x)
    x_i = tf.imag(x)

    with tf.variable_scope('c0'):
        conv0_r, conv0_i = myconv3d(x_r, x_i, [IMAGE_FRAMES, 9, 9, IMAGE_CHANNELS, KERNEL_NUM[0]], strides=2)
        bn0_r = batch_normalization(conv0_r, KERNEL_NUM[0], is_training, "_r")
        bn0_i = batch_normalization(conv0_i, KERNEL_NUM[0], is_training, "_i")
        relu0_r = tf.nn.relu(bn0_r)
        relu0_i = tf.nn.relu(bn0_i)

    # with tf.variable_scope('c1'):
    #     conv1 = myconv3d(relu0, [IMAGE_FRAMES, KERNEL_SIZE, KERNEL_SIZE, IMAGE_CHANNELS, KERNEL_NUM[0]])
    #     bn1 = batch_normalization(conv1, KERNEL_NUM[0], is_training)
    #     relu1 = tf.nn.relu(bn1)

    block_r = relu0_r
    block_i = relu0_i
    name = [['block1c1', 'block1c2'], ['block2c1', 'block2c2'], ['block3c1', 'block3c2'], ['block4c1', 'block4c2']]
    for block_id in range(len(BLOCK_LIST)):  # 第几个Block
        for layer_id in range(BLOCK_LIST[block_id]):  # 第几个卷积层
            with tf.variable_scope(name[block_id][layer_id]):
                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block_r, block_i = basicblock(block_r, block_i, DEPTH[block_id][layer_id], IMAGE_FRAMES, KERNEL_SIZE, KERNEL_NUM[block_id], is_training, strides=2,
                                       residual_path=True)
                else:
                    block_r, block_i = basicblock(block_r, block_i, DEPTH[block_id][layer_id], IMAGE_FRAMES, KERNEL_SIZE, KERNEL_NUM[block_id], is_training)
    return tf.complex(block_r, block_i)


# 定义前向传播函数
# 初始化各层的权重，设定计算式（计算图只规定计算式，不进行真实计算）
def resnet_forward2(x, IMAGE_FRAMES, KERNEL_SIZE, IMAGE_CHANNELS, is_training=True):
    # IMAGE_CHANNELS = 16  # 输入图片的深度
    # KERNEL_SIZE = 3  # 卷积核的大小（所有卷积核的大小都一样）
    # IMAGE_FRAMES = 10  # 输入图片的帧数
    # KERNEL_NUM = [64, 128, 256, 512]  # 残差模块各个block卷积核的个数
    # DEPTH = [[64, 64], [64, 128], [128, 256], [256, 512]]  # 残差模块各个block输入图片的深度
    KERNEL_NUM = [64, 128, 64, 8]  # 残差模块各个block卷积核的个数
    DEPTH = [[64, 64], [64, 128], [128, 64], [64, 8]]  # 残差模块各个block输入图片的深度
    BLOCK_LIST = [2, 2, 2, 2]  # 残差模块各个block卷积层的数量
    OUTPUT_SIZE = 3

    x_r = tf.real(x)
    x_i = tf.imag(x)

    with tf.variable_scope('c2'):
        conv1_r, conv1_i = myconv3d(x_r, x_i, [IMAGE_FRAMES, KERNEL_SIZE, KERNEL_SIZE, IMAGE_CHANNELS, KERNEL_NUM[0]])
        bn1_r = batch_normalization(conv1_r, KERNEL_NUM[0], is_training, "_r")
        bn1_i = batch_normalization(conv1_i, KERNEL_NUM[0], is_training, "_i")
        relu1_r = tf.nn.relu(bn1_r)
        relu1_i = tf.nn.relu(bn1_i)

    block_r = relu1_r
    block_i = relu1_i
    name = [['block1c12', 'block1c22'], ['block2c12', 'block2c22'], ['block3c12', 'block3c22'], ['block4c12', 'block4c22']]
    for block_id in range(len(BLOCK_LIST)):  # 第几个Block
        for layer_id in range(BLOCK_LIST[block_id]):  # 第几个卷积层
            with tf.variable_scope(name[block_id][layer_id]):
                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block_r, block_i = basicblock(block_r, block_i, DEPTH[block_id][layer_id], IMAGE_FRAMES, KERNEL_SIZE, KERNEL_NUM[block_id], is_training,
                                       strides=2,
                                       residual_path=True)
                else:
                    block_r, block_i = basicblock(block_r, block_i, DEPTH[block_id][layer_id], IMAGE_FRAMES, KERNEL_SIZE, KERNEL_NUM[block_id], is_training)

    # pool = tf.nn.avg_pool(block, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding="VALID")  # 全局平均池化

    pool_shape = block_r.get_shape().as_list()  # 得到各维度值
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] * pool_shape[4]
    reshaped_r = tf.reshape(block_r, [pool_shape[0], nodes])  # 拉直成一维向量输入全连接层
    reshaped_i = tf.reshape(block_i, [pool_shape[0], nodes])  # 拉直成一维向量输入全连接层

    with tf.variable_scope('f1'):
        y_r, y_i = fc(reshaped_r, reshaped_i, [nodes, OUTPUT_SIZE])

    return tf.complex(y_r, y_i)

