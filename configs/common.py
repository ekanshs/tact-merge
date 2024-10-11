"""Default Hyperparameter configuration to train on CIFAR10."""
import os
from typing import Sequence
import jax
import ml_collections



def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 0
  config.model = None
  config.dataset = None
  config.datasets = None
  config.label_smoothing_factor = 0.0
  config.permutations_dir = None
  config.pretrained_dir = None
  config.from_pretrained = True
  config.optimizer = None
  config.training_schedule = None

  # Shuffle buffer size.
  config.shuffle_buffer = 50_000
  # Run prediction on validation set every so many steps
  config.eval_every = 1_000
  # Prefetch data configs
  config.prefetch = 10 
  # Log progress every so many steps.
  config.progress_every = 100
  # How often to write checkpoints. Specifying 0 disables checkpointing.
  config.checkpoint_every = 1_000
  config.keep = None
  config.keep_every = None  
  
  config.half_precision = False

  config.run_id = None
  config.overwrite_cache = False
  config.num_workers = 1

  config.width_multiplier = 1
  config.depth_multiplier = 1.
  config.num_attention_heads = -1
  
  config.keep_batch_stats = False
  ## Trainable param blocks
  config.train_classifier_at_init = None  
  config.local_init = None  
  config.task_a_init = None  
  config.task_b_init = None  

  return config.lock()


TRAINING_SCHEDULE = {
  'cosine': ml_collections.ConfigDict({
    'decay_schedule': 'cosine',
    'warmup_steps': 1_000,
    'num_steps': 10_000,
    'per_device_train_batch_size': 128,
    'per_device_eval_batch_size': 128,
    'seed':42,
    'classifier' : ml_collections.ConfigDict({
      'num_steps': 5_000,
      'warmup_steps' : 5_00
    })
  }),
  'linear': ml_collections.ConfigDict({
    'decay_schedule': 'cosine',
    'warmup_steps': 1_000,
    'num_steps': 10_000,
    'per_device_train_batch_size': 128,
    'per_device_eval_batch_size': 128,
    'seed':42,
    'classifier' : ml_collections.ConfigDict({
      'num_steps': 5_000,
      'warmup_steps' : 5_00
    })
  }),

  'piecewise-constant': ml_collections.ConfigDict({
    'decay_schedule': 'piecewise-constant',
    'warmup_steps': 1_000,
    'num_steps': 10_000,
    'per_device_train_batch_size': 128,
    'per_device_eval_batch_size': 128,
    'gamma': 0.1,
    'milestones': (6_0000,8_000,9_000),
    'seed':42,
    'classifier' : ml_collections.ConfigDict({
      'num_steps': 5_000,
      'warmup_steps' : 500
    })
  }),
}



TRAIN_OPTIMIZER_PRESETS =  {  
    'sgd': ml_collections.ConfigDict(
      {'name': 'sgd',
        'learning_rate': 1e-3, 
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'clip_global_norm' : 100., 
        'classifier' : ml_collections.ConfigDict({'name': 'sgd',
          'learning_rate': 1e-3, 
          'momentum': 0.9,
          'weight_decay': 1e-1,
          'clip_global_norm' : 100., 
          'b1': 0.9,  
          'b2': 0.999,  
          'eps':1e-8,
          }),
        }),
    'adam': ml_collections.ConfigDict(
      {'name': 'adam',
        'learning_rate': 1e-5,
        'b1': 0.9,  
        'b2': 0.999,  
        'eps':1e-8,
        'weight_decay': 1e-1,
        'clip_global_norm' : 1.,
        'classifier' : ml_collections.ConfigDict({'name': 'sgd',
          'learning_rate': 1e-3, 
          'momentum': 0.9,
          'weight_decay': 1e-1,
          'clip_global_norm' : 100., 
          'b1': 0.9,  
          'b2': 0.999,  
          'eps':1e-8,
          }),
        }),    
    }


DATASET_PRESETS = ml_collections.ConfigDict({
  'cifar10': ml_collections.ConfigDict(
      { 
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
          'validation': 'train[:10%]',
          'test': 'test',        
          'crop': 224})
  }),
  'cifar100': ml_collections.ConfigDict(
      { 
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
              'validation': 'train[:10%]',
              'test': 'test',
              'crop': 224})
      }),
  'food101': ml_collections.ConfigDict(
      { 
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
          'validation': 'train[:10%]',
          'test': 'validation',
          'crop': 224})
      }),
  'eurosat': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train[:90%]',
            'mtl-train': 'train[:90%]',
            'validation': 'train[:9%]',
              'test': 'train[90%:]',
              'crop': 224})
      }),
  'svhn_cropped': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'train[:10%]',
            'test': 'test',
            'crop': 224})
      }),
  'sun397': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'train[:10%]',
            'test': 'test',
              'crop': 224})
      }),
  'imagenet2012': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
      'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'train[:10%]',
            'test': 'validation',
            'crop': 224})
      } ),
  'imagenette': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'train[:10%]',
              'test': 'validation',
              'crop': 224})
      } ),
  'stl10': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'train[:10%]',
              'test': 'test',
              'crop': 224})
      } ),
  'dtd': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'validation',
            'test': 'test',
            'crop': 224})
      } ),
    'gtsrb': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'validation',
            'test': 'test',
            'crop': 224})
      } ),
    'omniglot': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'validation',
            'test': 'test',
            'crop': 224})
      } ),
    'vgg-flowers': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'validation',
            'test': 'test',
            'crop': 224})
      } ),
    'aircraft': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'validation',
            'test': 'test',
            'crop': 224})
      } ), 
    'daimlerpedcls': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'validation',
            'test': 'test',
            'crop': 224})
      } ), 
    
  'ucf101': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
            'validation': 'validation',
            'test': 'test',
            'crop': 224})
      } ),
  'resisc45': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train[10%:90%]',
          'mtl-train': 'train[10%:90%]',
            'validation': 'train[:10%]',
              'test': 'train[90%:]',
              'crop': 224})
      } ),
    'caltech101': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
          'validation': 'test[90%:]',
          'test': 'test[:90%]',
          'crop': 224})
      } ),
    'cassava': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
          'validation': 'validation',
          'test': 'test',
          'crop': 224})
      } ),
    'oxford_iiit_pet': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
          'validation': 'test[90%:]',
          'test': 'test[:90%]',
          'crop': 224})
      } ),
    'oxford_flowers102': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
          'validation': 'validation',
          'test': 'test',
          'crop': 224})
      } ),
      'colorectal_histology': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train[:90%]',
          'mtl-train': 'train[:90%]',
          'validation': 'train[90%:]',
          'test': 'train[90%:]',
          'crop': 224})
      } ),
    'places365_small': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
          'validation': 'validation',
          'test': 'test',
          'crop': 224})
      } ),
    'stanford_dogs': ml_collections.ConfigDict(
      {
        'model_dir': '.',
        'init_model_dir': '.',
        'tfds_data_dir': None,
        'pp': ml_collections.ConfigDict(
          {'train': 'train',
          'mtl-train': 'train',
          'validation': 'test[90%:]',
          'test': 'test[:90%]',
          'crop': 224})
      } ),
  })
