#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import tensorflow as tf

HEIGHT = 256
WIDTH = 256
DEPTH = 3

class MyCustomDataSet(object):

  def __init__(self, data_dir, subset='train', use_distortion=True):
    self.data_dir = data_dir
    self.subset = subset
    self.use_distortion = use_distortion

  def get_filenames(self):
    if self.subset in ['train', 'validation', 'eval']:
      return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.image.decode_image(features['img_raw'], channels=3)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.reshape(image, [WIDTH, HEIGHT, DEPTH]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)

    # Custom preprocessing.
    label = tf.cast(features['label'], tf.int32)

    # Custom preprocessing.
    image = self.preprocess(image)

    return image, label

  def make_batch(self, batch_size):
    """Read the images and labels from 'filenames'."""
    filenames = self.get_filenames()
    # Repeat infinitely.
    dataset = tf.contrib.data.TFRecordDataset(filenames).repeat()

    # Parse records.
    dataset = dataset.map(
        self.parser, num_threads=batch_size, output_buffer_size=2 * batch_size)

    # Potentially shuffle records.
    if self.subset == 'train':
      min_queue_examples = int(MyCustomDataSet.num_examples_per_epoch(self.subset) * 0.1)
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling.
      dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    # Batch it up.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

  def preprocess(self, image):
    """Preprocess a single image in [height, width, depth] layout."""
    image = tf.image.per_image_standardization(image)
    if self.subset == 'train' and self.use_distortion:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
      image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
      image = tf.image.random_flip_left_right(image)
    return image

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 2930010
    elif subset == 'validation':
      return 200000
    elif subset == 'eval':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)
