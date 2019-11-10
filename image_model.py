#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

class MyCustomModel(object):

  def __init__(self,
               is_training,
               batch_norm_decay,
               batch_norm_epsilon,
               dropout_rate = 0.5,
               data_format='channels_first'):
    # Add one in case label starts with 1. No impact if label starts with 0.
    self.num_classes = 2

    self._is_training = is_training
    self._batch_norm_decay = batch_norm_decay
    self._batch_norm_epsilon = batch_norm_epsilon
    self._dropout_rate = dropout_rate
    assert data_format in ('channels_first', 'channels_last')
    self._data_format = data_format

  def forward_pass(self, x, input_data_format='channels_last'):
    """Build the core model within the graph."""
    if self._data_format != input_data_format:
      if input_data_format == 'channels_last':
        # Computation requires channels_first.
        x = tf.transpose(x, [0, 3, 1, 2])
      else:
        # Computation requires channels_last.
        x = tf.transpose(x, [0, 2, 3, 1])

    # 512 * 512 * 3

    # Use basic (non-bottleneck) block and ResNet V2.
    res_func = self._residual_v2

    x = self._conv(x, 7, 32, 1)
    x = self._max_pool(x, 3, 2)

    x = res_func(x, 3, 32, 64, 2)
    x = res_func(x, 3, 64, 64, 1)
    x = res_func(x, 3, 64, 64, 1)
    x = self._batch_norm(x)

    x = res_func(x, 3, 64, 128, 2)
    x = res_func(x, 3, 128, 128, 1)
    x = res_func(x, 3, 128, 128, 1)
    x = self._batch_norm(x)

    x = res_func(x, 3, 128, 256, 2)
    x = res_func(x, 3, 256, 256, 1)
    x = res_func(x, 3, 256, 256, 1)
    x = self._batch_norm(x)

    x = res_func(x, 3, 256, 512, 2)
    x = res_func(x, 3, 512, 512, 1)
    x = res_func(x, 3, 512, 512, 1)
    x = self._batch_norm(x)


    x = self._fully_connected_relu(x, 2048)
    x = self._fully_connected_relu(x, 2048)
    x = self._dropout(x, self._dropout_rate)

    x = self._fully_connected(x, self.num_classes)

    return x

  def _residual_v1(self,
                   x,
                   kernel_size,
                   in_filter,
                   out_filter,
                   stride,
                   activate_before_residual=False):
    """Residual unit with 2 sub layers, using Plan A for shortcut connection."""

    del activate_before_residual
    with tf.name_scope('residual_v1') as name_scope:
      orig_x = x

      x = self._conv(x, kernel_size, out_filter, stride)
      x = self._batch_norm(x)
      x = self._relu(x)

      x = self._conv(x, kernel_size, out_filter, 1)
      x = self._batch_norm(x)

      if in_filter != out_filter:
        orig_x = self._avg_pool(orig_x, stride, stride)
        pad = (out_filter - in_filter) // 2
        if self._data_format == 'channels_first':
          orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        else:
          orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

      x = self._relu(tf.add(x, orig_x))

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _residual_v2(self,
                   x,
                   in_filter,
                   out_filter,
                   stride,
                   activate_before_residual=False):
    """Residual unit with 2 sub layers with preactivation, plan A shortcut."""

    with tf.name_scope('residual_v2') as name_scope:
      if activate_before_residual:
        x = self._batch_norm(x)
        x = self._relu(x)
        orig_x = x
      else:
        orig_x = x
        x = self._batch_norm(x)
        x = self._relu(x)

      x = self._conv(x, 3, out_filter, stride)

      x = self._batch_norm(x)
      x = self._relu(x)
      x = self._conv(x, 3, out_filter, [1, 1, 1, 1])

      if in_filter != out_filter:
        pad = (out_filter - in_filter) // 2
        orig_x = self._avg_pool(orig_x, stride, stride)
        if self._data_format == 'channels_first':
          orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        else:
          orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

      x = tf.add(x, orig_x)

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _bottleneck_residual_v2(self,
                              x,
                              in_filter,
                              out_filter,
                              stride,
                              activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers, plan B shortcut."""

    with tf.name_scope('bottle_residual_v2') as name_scope:
      if activate_before_residual:
        x = self._batch_norm(x)
        x = self._relu(x)
        orig_x = x
      else:
        orig_x = x
        x = self._batch_norm(x)
        x = self._relu(x)

      x = self._conv(x, 1, out_filter // 4, stride, is_atrous=True)

      x = self._batch_norm(x)
      x = self._relu(x)
      # pad when stride isn't unit
      x = self._conv(x, 3, out_filter // 4, 1, is_atrous=True)

      x = self._batch_norm(x)
      x = self._relu(x)
      x = self._conv(x, 1, out_filter, 1, is_atrous=True)

      if in_filter != out_filter:
        orig_x = self._conv(orig_x, 1, out_filter, stride, is_atrous=True)
      x = tf.add(x, orig_x)

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _conv(self, x, kernel_size, filters, strides, is_atrous=False):
    """Convolution."""

    padding = 'SAME'
    if not is_atrous and strides > 1:
      pad = kernel_size - 1
      pad_beg = pad // 2
      pad_end = pad - pad_beg
      if self._data_format == 'channels_first':
        x = tf.pad(x, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
      else:
        x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
      padding = 'VALID'
    return tf.layers.conv2d(
        inputs=x,
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=self._data_format)

  def _conv_relu(self, x, kernel_size, filters, strides, is_atrous=False):
    """Convolution."""

    padding = 'SAME'
    if not is_atrous and strides > 1:
      pad = kernel_size - 1
      pad_beg = pad // 2
      pad_end = pad - pad_beg
      if self._data_format == 'channels_first':
        x = tf.pad(x, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
      else:
        x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
      padding = 'VALID'
    return tf.layers.conv2d(
        inputs=x,
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        padding=padding,
        use_bias=False,
        activation=tf.nn.relu,
        data_format=self._data_format)

  def _batch_norm(self, x):
    if self._data_format == 'channels_first':
      data_format = 'NCHW'
    else:
      data_format = 'NHWC'
    return tf.contrib.layers.batch_norm(
        x,
        decay=self._batch_norm_decay,
        center=True,
        scale=True,
        epsilon=self._batch_norm_epsilon,
        is_training=self._is_training,
        fused=True,
        data_format=data_format)

  def _relu(self, x):
    return tf.nn.relu(x)

  def _fully_connected(self, x, out_dim):
    with tf.name_scope('fully_connected') as name_scope:
      x = tf.layers.dense(x, out_dim)

    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _fully_connected_relu(self, x, out_dim):
    with tf.name_scope('fully_connected') as name_scope:
      x = tf.layers.dense(x, out_dim, activation=tf.nn.relu)

    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _dropout(self, x, rate):
    with tf.name_scope('dropout') as name_scope:
      x = tf.layers.dropout(x, rate, training=self._is_training)

  def _avg_pool(self, x, pool_size, stride):
    with tf.name_scope('avg_pool') as name_scope:
      x = tf.layers.average_pooling2d(
          x, pool_size, stride, 'SAME', data_format=self._data_format)

    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _max_pool(self, x, pool_size, stride):
    with tf.name_scope('avg_pool') as name_scope:
      x = tf.layers.max_pooling2d(
          x, pool_size, stride, 'SAME', data_format=self._data_format)

    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _global_avg_pool(self, x):
    with tf.name_scope('global_avg_pool') as name_scope:
      assert x.get_shape().ndims == 4
      if self._data_format == 'channels_first':
        x = tf.reduce_mean(x, [2, 3])
      else:
        x = tf.reduce_mean(x, [1, 2])
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x