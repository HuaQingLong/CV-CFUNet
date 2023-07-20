# @Time    : 2021/4/8 14:30
# @Author  : HuaQinglong
# @Email   : hqlhit2014@163.com
# @File    : CVConvLSTMCell.py
# package version:
#               python 3.6
#               numpy 1.19.4
#               tensorflow 1.14.0
# 复数域卷积长短时记忆网络单元

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf

class BasicConvLSTM(rnn_cell_impl.RNNCell):
    def __init__(self,
                 conv_ndims,
                 input_shape,
                 output_channels,
                 kernel_shape,
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=1.0,
                 peephole=False,
                 initializers=None,
                 name="conv_lstm_cell"):
        """Construct ConvLSTMCell.
        Args:
          conv_ndims: Convolution dimensionality (1, 2 or 3).
          input_shape: Shape of the input as int tuple, excluding the batch size.
          output_channels: int, number of output channels of the conv LSTM.
          kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
          use_bias: Use bias in convolutions.
          skip_connection: If set to `True`, concatenate the input to the
          output of the conv LSTM. Default: `False`.
          forget_bias: Forget bias.
          name: Name of the module.
        Raises:
          ValueError: If `skip_connection` is `True` and stride is different from 1
            or if `input_shape` is incompatible with `conv_ndims`.
        """
        super(BasicConvLSTM, self).__init__(name=name)

        if conv_ndims != len(input_shape) - 1:
            raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
                input_shape, conv_ndims))

        self._conv_ndims = conv_ndims
        self._input_shape = input_shape
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._use_bias = use_bias
        self._forget_bias = forget_bias
        self._skip_connection = skip_connection
        self._peephole = peephole

        self._total_output_channels = output_channels
        if self._skip_connection:
            self._total_output_channels += self._input_shape[-1]

        state_size = tensor_shape.TensorShape(
            self._input_shape[:-1] + [self._output_channels])
        self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
        self._output_size = tensor_shape.TensorShape(self._input_shape[:-1]
                                                     + [self._total_output_channels])

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def call(self, inputs, state, scope=None):
        inputs_r = tf.real(inputs)
        inputs_i = tf.imag(inputs)

        hidden = state
        hidden_r = tf.real(hidden)
        hidden_i = tf.imag(hidden)

        new_hidden_r, new_hidden_i = _conv([hidden_r],
                                           [hidden_i],
                                           self._kernel_shape,
                                           3 * self._output_channels,
                                           self._use_bias,
                                           scopname="_hidden")

        new_inputs_r, new_inputs_i = _conv([inputs_r],
                                           [inputs_i],
                                           self._kernel_shape,
                                           3 * self._output_channels,
                                           self._use_bias,
                                           scopname="_inputs")

        gates_hidden_r = array_ops.split(value=new_hidden_r,
                                num_or_size_splits=3,
                                axis=self._conv_ndims + 1)

        gates_hidden_i = array_ops.split(value=new_hidden_i,
                                  num_or_size_splits=3,
                                  axis=self._conv_ndims + 1)

        hz_gate_r, hr_gate_r, hh_gate_r = gates_hidden_r
        hz_gate_i, hr_gate_i, hh_gate_i = gates_hidden_i

        gates_inputs_r = array_ops.split(value=new_inputs_r,
                                       num_or_size_splits=3,
                                       axis=self._conv_ndims + 1)

        gates_inputs_i = array_ops.split(value=new_inputs_i,
                                         num_or_size_splits=3,
                                         axis=self._conv_ndims + 1)

        xz_gate_r, xr_gate_r, xh_gate_r = gates_inputs_r
        xz_gate_i, xr_gate_i, xh_gate_i = gates_inputs_i

        update_gate_r = math_ops.sigmoid(xz_gate_r + hz_gate_r)
        update_gate_i = math_ops.sigmoid(xz_gate_i + hz_gate_i)

        reset_gate_r = math_ops.sigmoid(xr_gate_r + hr_gate_r)
        reset_gate_i = math_ops.sigmoid(xr_gate_i + hr_gate_i)

        mid_hidden_r = math_ops.tanh(xh_gate_r + reset_gate_r * hh_gate_r - reset_gate_i * hh_gate_i)
        mid_hidden_i = math_ops.tanh(xh_gate_i + reset_gate_r * hh_gate_i + reset_gate_i * hh_gate_r)

        # output_r = (1 - update_gate_r) * mid_hidden_r - (1 - update_gate_i) * mid_hidden_i + update_gate_r * hidden_r - update_gate_i * hidden_i
        # output_i = (1 - update_gate_r) * mid_hidden_i + (1 - update_gate_i) * mid_hidden_r + update_gate_r * hidden_i + update_gate_i * hidden_r

        # output_r = mid_hidden_r - update_gate_r * mid_hidden_r + update_gate_i * mid_hidden_i + update_gate_r * hidden_r - update_gate_i * hidden_i
        # output_i = mid_hidden_i - update_gate_r * mid_hidden_i - update_gate_i * mid_hidden_r + update_gate_r * hidden_i + update_gate_i * hidden_r

        # output_r = update_gate_r * mid_hidden_r + update_gate_i * mid_hidden_i + update_gate_r * hidden_r - update_gate_i * hidden_i
        # output_i = update_gate_r * mid_hidden_i - update_gate_i * mid_hidden_r + update_gate_r * hidden_i + update_gate_i * hidden_r

        # output_r = update_gate_i * mid_hidden_r - update_gate_r * mid_hidden_i + update_gate_r * hidden_r - update_gate_i * hidden_i
        # output_i = update_gate_i * mid_hidden_i + update_gate_r * mid_hidden_r + update_gate_r * hidden_i + update_gate_i * hidden_r

        output_r = (1 - update_gate_r) * mid_hidden_r + update_gate_r * hidden_r
        output_i = (1 - update_gate_i) * mid_hidden_i + update_gate_i * hidden_i

        output = tf.complex(output_r, output_i)
        return output, output


def _conv(args_r, args_i, filter_size, num_features, bias, bias_start=0.0, scopname=''):
    """convolution:
    Args:
      args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
      batch x n, Tensors.
      filter_size: int tuple of filter height and width.
      num_features: int, number of features.
      bias_start: starting value to initialize the bias; 0 by default.
    Returns:
      A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args_r]
    shape_length = len(shapes[0])
    for shape in shapes:
        if len(shape) not in [3, 4, 5]:
            raise ValueError("Conv Linear expects 3D, 4D "
                             "or 5D arguments: %s" % str(shapes))
        if len(shape) != len(shapes[0]):
            raise ValueError("Conv Linear expects all args "
                             "to be of same Dimension: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[-1]
    dtype = [a.dtype for a in args_r][0]

    # determine correct conv operation
    if shape_length == 3:
        conv_op = nn_ops.conv1d
        strides = 1
    elif shape_length == 4:
        conv_op = nn_ops.conv2d
        strides = shape_length * [1]
    elif shape_length == 5:
        conv_op = nn_ops.conv3d
        strides = shape_length * [1]

    # Now the computation.
    # kernel_r = vs.get_variable("kernel_r"+scopname, filter_size + [total_arg_size_depth, num_features], dtype=dtype)
    # kernel_i = vs.get_variable("kernel_i"+scopname, filter_size + [total_arg_size_depth, num_features], dtype=dtype)
    kernel_r = vs.get_variable("kernel_r" + scopname,
                               filter_size + [total_arg_size_depth, num_features],
                               dtype=dtype,
                               initializer=tf.contrib.layers.xavier_initializer_conv2d())
    kernel_i = vs.get_variable("kernel_i" + scopname,
                               filter_size + [total_arg_size_depth, num_features],
                               dtype=dtype,
                               initializer=tf.contrib.layers.xavier_initializer_conv2d())

    if len(args_r) == 1:
        res_r = conv_op(args_r[0], kernel_r, strides, padding='SAME') - conv_op(args_i[0], kernel_i, strides, padding='SAME')
        res_i = conv_op(args_r[0], kernel_i, strides, padding='SAME') + conv_op(args_i[0], kernel_r, strides, padding='SAME')
    else:
        res_r = conv_op(array_ops.concat(axis=shape_length - 1, values=args_r), kernel_r, strides, padding='SAME') \
                - conv_op(array_ops.concat(axis=shape_length - 1, values=args_i), kernel_i, strides, padding='SAME')
        res_i = conv_op(array_ops.concat(axis=shape_length - 1, values=args_r), kernel_i, strides, padding='SAME') \
                + conv_op(array_ops.concat(axis=shape_length - 1, values=args_i), kernel_r, strides, padding='SAME')
    if not bias:
        return res_r, res_i
    bias_term_r = vs.get_variable("biases_r"+scopname, [num_features], dtype=dtype,
                                initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    bias_term_i = vs.get_variable("biases_i"+scopname, [num_features], dtype=dtype,
                                initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return res_r + bias_term_r, res_i + bias_term_i