# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os

from absl import logging
import flax
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


import sys
if sys.platform != 'darwin':
  # A workaround to avoid crash because tfds may open to many files.
  import resource
  low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

tf.config.experimental.set_visible_devices([], 'GPU')



def get_tfds_info(dataset, split):
  """Returns information about tfds dataset -- see `get_dataset_info()`."""
  data_builder = tfds.builder(dataset)
  if dataset == 'sun397':
    int2str = lambda idx: data_builder.info.features['label'].int2str(idx).split('/')[2] \
                      if len(data_builder.info.features['label'].int2str(idx).split('/')) == 3 else \
                      data_builder.info.features['label'].int2str(idx).split('/')[3] + " " + data_builder.info.features['label'].int2str(idx).split('/')[2] 
  else:
    int2str = data_builder.info.features['label'].int2str
  
  return dict(
      num_examples=data_builder.info.splits[split].num_examples,
      num_classes=data_builder.info.features['label'].num_classes,
      int2str=int2str,
      examples_glob=None,
  )


def get_directory_info(directory):
  """Returns information about directory dataset -- see `get_dataset_info()`."""
  examples_glob = f'{directory}/*/*.jpg'
  paths = glob.glob(examples_glob)
  get_classname = lambda path: path.split('/')[-2]
  class_names = sorted(set(map(get_classname, paths)))
  return dict(
      num_examples=len(paths),
      num_classes=len(class_names),
      int2str=lambda id_: class_names[id_],
      examples_glob=examples_glob,
  )


def get_dataset_info(dataset, split):
  """Returns information about a dataset.

  Args:
    dataset: Name of tfds dataset or directory -- see `./configs/common.py`
    split: Which split to return data for (e.g. "test", or "train"; tfds also
      supports splits like "test[:90%]").

  Returns:
    A dictionary with the following keys:
    - num_examples: Number of examples in dataset/mode.
    - num_classes: Number of classes in dataset.
    - int2str: Function converting class id to class name.
    - examples_glob: Glob to select all files, or None (for tfds dataset).
  """
  directory = os.path.join(dataset, split)
  if os.path.isdir(directory):
    return get_directory_info(directory)
  return get_tfds_info(dataset, split)


def get_datasets(config, dataset, repeats=None, batch_size=None):
  """Returns `ds_train, ds_test` for specified `config`."""

  if os.path.isdir(dataset):
    train_dir = os.path.join(dataset, 'train')
    test_dir = os.path.join(dataset, 'test')
    if not os.path.isdir(train_dir):
      raise ValueError('Expected to find directories"{}" and "{}"'.format(
          train_dir,
          test_dir,
      ))
    logging.info('Reading dataset from directories "%s" and "%s"', train_dir, test_dir)
    ds_train = get_data_from_directory(config=config, dataset=dataset, directory=train_dir, mode='train', repeats=repeats, batch_size=batch_size)
    ds_test = get_data_from_directory(config=config, dataset=dataset, directory=test_dir, mode='test', repeats=repeats, batch_size=batch_size)
  else:
    logging.info('Reading dataset from tfds "%s"', dataset)
    ds_train = get_data_from_tfds(config=config, dataset=dataset, mode='train', repeats=repeats, batch_size=batch_size)
    ds_test = get_data_from_tfds(config=config, dataset=dataset, mode='test', repeats=repeats, batch_size=batch_size)
    
  return ds_train, ds_test


def get_unlabelled_datasets(config, datasets, split='train[:1000]', batch_size=None, repeats=None):
  if batch_size == None:
    batch_size = config.training_schedule.per_device_train_batch_size * jax.device_count()
  dataset = datasets[0]
  image_size = config[dataset].pp['crop']
  data_builder = tfds.builder(dataset, data_dir=config[dataset].tfds_data_dir)
  data_builder.download_and_prepare(
    download_config=tfds.download.DownloadConfig(
        manual_dir=config[dataset].tfds_data_dir))
  
  @tfds.decode.make_decoder()
  def decode_example(serialized_image, features):
    im =  data_builder.info.features['image'].decode_example(serialized_image)
    return im

  data = data_builder.as_dataset(
      split=split,
      decoders={'image': decode_example()},
      as_supervised=False,
      shuffle_files=True)
  
  data = data.map(lambda d: {'image':d['image']})
  shuffle_buffer = get_tfds_info(dataset, split)['num_examples']
  
  for dataset in datasets[1:]:
    data_builder = tfds.builder(dataset, data_dir=config[dataset].tfds_data_dir)
    data_builder.download_and_prepare(
      download_config=tfds.download.DownloadConfig(
          manual_dir=config[dataset].tfds_data_dir))
    
    @tfds.decode.make_decoder()
    def decode_example(serialized_image, features):
      im =  data_builder.info.features['image'].decode_example(serialized_image)
      return im

    data = data.concatenate(data_builder.as_dataset(
        split=split,
        decoders={'image': decode_example()},
        as_supervised=False,
        shuffle_files=True).map(lambda d: {'image':d['image']}))
    shuffle_buffer += get_tfds_info(dataset, split)['num_examples']
  
  return get_unlabelled_data(
      data=data,
      batch_size=batch_size,
      image_size=image_size,
      shuffle_buffer=shuffle_buffer, 
      seed=config.seed, 
      repeats=repeats)


def get_unlabelled_dataset(config, dataset, split='test', batch_size=None, repeats = 1):
  if batch_size == None:
    batch_size = config.training_schedule.per_device_train_batch_size * jax.device_count()
  image_size = config[dataset].pp['crop']
  data_builder = tfds.builder(dataset, data_dir=config[dataset].tfds_data_dir)
  data_builder.download_and_prepare(
    download_config=tfds.download.DownloadConfig(
        manual_dir=config[dataset].tfds_data_dir))
  
  @tfds.decode.make_decoder()
  def decode_example(serialized_image, features):
    im =  data_builder.info.features['image'].decode_example(serialized_image)
    return im

  data = data_builder.as_dataset(
      split=config[dataset].pp[split],
      decoders={'image': decode_example()},
      as_supervised=False,
      shuffle_files=True)

  shuffle_buffer = get_tfds_info(dataset, config[dataset].pp[split])['num_examples']
    
  return get_unlabelled_data(
      data=data,
      batch_size=batch_size,
      image_size=image_size,
      shuffle_buffer=shuffle_buffer, 
      repeats=repeats,
      seed=config.seed)



def get_datasets_with_validation(config, dataset, repeats=None, batch_size=None):
  """Returns `ds_train, ds_test` for specified `config`."""
  ds_train, ds_test = get_datasets(config, dataset, repeats=repeats)
  
  logging.info('Reading dataset from tfds "%s"', dataset)
  ds_val = get_data_from_tfds(config=config, dataset=dataset, mode='validation', repeats=None)
  
  return ds_train, ds_test, ds_val

def get_datasets_for_mtl(config, datasets, repeats=None, batch_size=None):
  """Returns `ds_train, ds_test` for specified `config`."""
  ds_train_ls = []
  ds_test_ls = []
  for dataset in datasets:
    logging.info('Reading dataset from tfds "%s"', dataset)
    ds_train_ls += [get_data_from_tfds(config=config, dataset=dataset, mode='train', repeats=repeats, batch_size=batch_size)]
    ds_test_ls += [get_data_from_tfds(config=config, dataset=dataset, mode='test', repeats=repeats, batch_size=batch_size)]
  return ds_train_ls, ds_test_ls

def get_data_from_directory(*, config, dataset, directory, mode, repeats=None, batch_size=None):
  """Returns dataset as read from specified `directory`."""
  if batch_size is None:
    batch_size=config.training_schedule.per_device_eval_batch_size * jax.device_count() if mode == 'test' else config.training_schedule.per_device_train_batch_size * jax.device_count()
  
  if repeats is None:
    repeats=1 if mode == 'test' else None
  
  dataset_info = get_directory_info(directory)
  data = tf.data.Dataset.list_files(dataset_info['examples_glob'])
  class_names = [
      dataset_info['int2str'](id_) for id_ in range(dataset_info['num_classes'])
  ]

  def _pp(path):
    return dict(
        image=path,
        label=tf.where(
            tf.strings.split(path, '/')[-2] == class_names
        )[0][0],
    )

  image_decoder = lambda path: tf.image.decode_jpeg(tf.io.read_file(path), 3)

  return get_data(
      data=data,
      mode=mode,
      num_classes=dataset_info['num_classes'],
      image_decoder=image_decoder,
      repeats=repeats,
      batch_size=batch_size,
      image_size=config[dataset].pp['crop'],
      shuffle_buffer=min(dataset_info['num_examples'], config.shuffle_buffer),
      preprocess=_pp, 
      seed=config.seed)


def get_data_from_tfds(*, config, dataset, mode, repeats=None, batch_size=None):
  """Returns dataset as read from tfds dataset `dataset`."""
  if batch_size is None:
    batch_size = config.training_schedule.per_device_eval_batch_size * jax.device_count() if mode == 'test' else config.training_schedule.per_device_train_batch_size * jax.device_count()
  
  data_builder = tfds.builder(dataset, data_dir=config[dataset].tfds_data_dir)

  data_builder.download_and_prepare(
      download_config=tfds.download.DownloadConfig(
          manual_dir=config[dataset].tfds_data_dir))
  data = data_builder.as_dataset(
      split=config[dataset].pp[mode],
      # Reduces memory footprint in shuffle buffer.
      decoders={'image': tfds.decode.SkipDecoding()},
      shuffle_files=(not(mode == 'test')))
  image_decoder = data_builder.info.features['image'].decode_example

  dataset_info = get_tfds_info(dataset, config[dataset].pp[mode])
  
  return get_data(
      data=data,
      mode=mode,
      num_classes=dataset_info['num_classes'],
      image_decoder=image_decoder,
      repeats= 1 if mode == 'test' else repeats,
      batch_size=batch_size,
      image_size=config[dataset].pp['crop'],
      shuffle_buffer=min(dataset_info['num_examples'], config.shuffle_buffer), 
      seed=config.seed)

def get_unlabelled_data(*,
             data,
             batch_size,
             image_size,
             shuffle_buffer,
             preprocess=None, 
             repeats=1,
             seed=0):
  def _pp(data):
    im = data['image']
    im = tf.image.resize(im, [image_size, image_size])
    im = (im - 127.5) / 127.5
    return {'image': im}

  data = data.repeat(repeats)
  
  data = data.shuffle(shuffle_buffer, seed=seed)
  
  if preprocess is not None:
    data = data.map(preprocess, tf.data.experimental.AUTOTUNE)
  
  data = data.map(_pp, tf.data.experimental.AUTOTUNE)
  data = data.batch(batch_size, drop_remainder=True)

  # Shard data such that it can be distributed across devices
  devices = jax.local_devices()
  num_devices = len(devices)
  
  def _shard(data):
    data['image'] = tf.reshape(data['image'],
                              [num_devices, -1, image_size, image_size,
                                data['image'].shape[-1]])
    return data

  # Shard data if num_devices > 1
  if num_devices > 1: 
    data = data.map(_shard, tf.data.experimental.AUTOTUNE)
    return data.prefetch(1)
  
  return data.prefetch(1)

def get_data(*,
             data,
             mode,
             num_classes,
             image_decoder,
             repeats,
             batch_size,
             image_size,
             shuffle_buffer,
             preprocess=None, 
             seed=0):
  """Returns dataset for training/eval.

  Args:
    data: tf.data.Dataset to read data from.
    mode: Must be "train" or "test".
    num_classes: Number of classes (used for one-hot encoding).
    image_decoder: Applied to `features['image']` after shuffling. Decoding the
      image after shuffling allows for a larger shuffle buffer.
    repeats: How many times the dataset should be repeated. For indefinite
      repeats specify None.
    batch_size: Global batch size. Note that the returned dataset will have
      dimensions [local_devices, batch_size / local_devices, ...].
    image_size: Image size after cropping (for training) / resizing (for
      evaluation).
    shuffle_buffer: Number of elements to preload the shuffle buffer with.
    preprocess: Optional preprocess function. This function will be applied to
      the dataset just after repeat/shuffling, and before the data augmentation
      preprocess step is applied.
  """

  def _pp(data):
    im = image_decoder(data['image'])
    if mode == 'train':
      channels = im.shape[-1]
      begin, size, _ = tf.image.sample_distorted_bounding_box(
          tf.shape(im),
          tf.zeros([0, 0, 4], tf.float32),
          area_range=(0.05, 1.0),
          min_object_covered=0,  # Don't enforce a minimum area.
          use_image_if_no_bounding_boxes=True)
      im = tf.slice(im, begin, size)
      # Unfortunately, the above operation loses the depth-dimension. So we
      # need to restore it the manual way.
      im.set_shape([None, None, channels])
      im = tf.image.resize(im, [image_size, image_size])
      if tf.random.uniform(shape=[]) > 0.5:
        im = tf.image.flip_left_right(im)
    else:
      im = tf.image.resize(im, [image_size, image_size])
    im = (im - 127.5) / 127.5
    label = tf.one_hot(data['label'], num_classes)  # pylint: disable=no-value-for-parameter
    return {'image': im, 'label': label}

  data = data.repeat(repeats)
  
  if not(mode == 'test'):
    data = data.shuffle(shuffle_buffer, seed=seed)
  
  if preprocess is not None:
    data = data.map(preprocess, tf.data.experimental.AUTOTUNE)
  data = data.map(_pp, tf.data.experimental.AUTOTUNE)
  data = data.batch(batch_size, drop_remainder=True)

  # Shard data such that it can be distributed across devices
  devices = jax.local_devices()
  num_devices = len(devices)
  
  def _shard(data):
    data['image'] = tf.reshape(data['image'],
                              [num_devices, -1, image_size, image_size,
                                data['image'].shape[-1]])
    data['label'] = tf.reshape(data['label'],
                              [num_devices, -1, num_classes])
    return data

  # Shard data if num_devices > 1
  if num_devices > 1: 
    data = data.map(_shard, tf.data.experimental.AUTOTUNE)
    return data.prefetch(1)
  
  return data.prefetch(1)

def _convert_to_numpy_iter(dataset):
  ds_iter = iter(dataset)
  ds_iter = map(lambda x: jax.tree_map(lambda t: jnp.asarray(memoryview(t)), x),
                ds_iter)
  return ds_iter

def _prefetch(dataset, n_prefetch):
  """Prefetches data to device and converts to numpy array."""
  if n_prefetch:
    dataset = dataset.prefetch(n_prefetch)
  ds_iter = _convert_to_numpy_iter(dataset)
  return ds_iter

def _shard_prefetch(dataset, n_prefetch):
  """Prefetches data sharded across devices and converts to numpy array."""
  ds_iter = _convert_to_numpy_iter(dataset)
  if n_prefetch:
    ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
  return ds_iter

def prefetch(dataset, n_prefetch, axis_name):
  """Prefetches data to device and converts to numpy array."""
  if axis_name is not None:
    return _shard_prefetch(dataset, n_prefetch)
  
  return _prefetch(dataset, n_prefetch)
