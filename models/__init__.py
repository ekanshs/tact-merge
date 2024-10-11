import copy
import jax
import jax.numpy as jnp
from flax import traverse_util

from transformers import FlaxAutoModel, FlaxViTForImageClassification

from .vgg import (VGG16 as VGG16, 
                  VGG19 as VGG19)


from .vit import (ViTmaeB16 as ViTmaeB16 ,
                  ViTmaeL14 as ViTmaeL14  ,
                  ViTB16 as ViTB16,
                  ViTL14 as ViTL14  ,
                  ViTB16IN21k as ViTB16IN21k  ,
                  ViTL14IN21k as ViTL14IN21k  ,
                  dinoViTB16 as dinoViTB16 , 
                  PermViTB16 as PermViTB16)

def create_model(*, model_cls, num_classes, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32

  return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)

def get_hf_pretrained_params(model, init, keep_batch_stats=False):
  if 'openai' in model.config.name_or_path:
    params = FlaxAutoModel.from_pretrained(model.config.name_or_path).params
  elif 'facebook' in model.config.name_or_path or 'dino' in model.config.name_or_path:
    params = FlaxViTForImageClassification.from_pretrained(model.config.name_or_path, from_pt=True).params
  else:
    params = FlaxViTForImageClassification.from_pretrained(model.config.name_or_path).params
  return model.load_pretrained_params(init, params, keep_batch_stats=keep_batch_stats)

def set_batch_norm_params_from_batch_stats(params, batch_stats):
  flat_params = traverse_util.flatten_dict(params)
  flat_batch_stats = traverse_util.flatten_dict(batch_stats)
  for k, v in flat_batch_stats.items():
    if k[-1] == 'mean':
      flat_params[('params',) + k[:-1] + ('bias',)] = v
    elif k[-1] == 'var':
      flat_params[('params',) + k[:-1] + ('scale',)] = jnp.sqrt(v)
  return traverse_util.unflatten_dict(flat_params)

def set_batch_norm_params(params, p):
  flat_params = traverse_util.flatten_dict(params)
  flat_p = traverse_util.flatten_dict(p)
  flat_bs = traverse_util.flatten_dict(p['batch_stats'])
  for k in flat_bs.keys():
      flat_params[('params',) + k[:-1] + ('bias',)] = flat_p[('params',) + k[:-1] + ('bias',)]
  return traverse_util.unflatten_dict(flat_params)



def load_batch_norm_params(target, source):
  flat_source = traverse_util.flatten_dict(source)
  flat_target = traverse_util.flatten_dict(target)
  batch_norm_candidates = ["batchnorm", "batch_norm", "bn", "normalization"] ## hugging-face resnet labels batch norm parameter as 'normalization'
  for k, v in flat_source.items():
    for bn_candidate in batch_norm_candidates:
      if bn_candidate in "".join(k).lower():
        flat_target[k] = copy.deepcopy(v)
  return traverse_util.unflatten_dict(flat_target)

def load_from_source(target, source):
  flat_source = traverse_util.flatten_dict(source)
  flat_target = traverse_util.flatten_dict(target)
  for k, v in flat_source.items():
    flat_target[k] = v
  return traverse_util.unflatten_dict(flat_target)

def reset_batch_stats(params):
  assert 'batch_stats' in list(params.keys())
  flat_batch_stats = traverse_util.flatten_dict(params['batch_stats'])
  flat_reset_batch_stats = {}
  for k, v in flat_batch_stats.items():
    if k[-1] == 'mean':
      flat_reset_batch_stats[k] = jnp.zeros_like(v)
    elif k[-1] == 'var':
      flat_reset_batch_stats[k] = jnp.ones_like(v)
    else:
      raise KeyError(f'Incorrect key ({k}) detected in batch_stats')
  params['batch_stats'] = traverse_util.unflatten_dict(flat_reset_batch_stats)
  return params
