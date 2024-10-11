from functools import partial

from typing import Any, Sequence

import jax.nn
import jax.numpy as jnp

from flax import linen as nn
from jax import random
from permutations import PermutationSpec, permutation_spec_from_axes_to_perm, conv_no_bias_axes_to_perm, dense_axes_to_perm, norm_axes_to_perm, dense_no_bias_axes_to_perm

from models.train_state import TrainState
from utils import flatten_params, unflatten_params, unfreeze, freeze
ModuleDef = Any

class VGGEncoder(nn.Module):
  """VGG Image Embedder."""
  backbone_layers: Any 
  classifier_width: int
  # num_classes: int
  norm: ModuleDef = nn.LayerNorm
  width_multiplier: int = 1
  tracker : bool = False
  repaired : bool = False
  
  @nn.compact
  def __call__(self, x, deterministic, output_every_layer = False):
    if output_every_layer:
      outs = []
    for l in self.backbone_layers:
      if isinstance(l, int):
        x = nn.Conv(features=l* self.width_multiplier, kernel_size=(3, 3), padding='same', use_bias=False)(x)
        x = self.norm()(x)
        if self.tracker == True:
          out = nn.BatchNorm()(x, use_running_average=deterministic)
        elif self.repaired == True:
          x = nn.BatchNorm()(x, use_running_average=deterministic)
        if output_every_layer: 
          outs += [x]
        x = nn.relu(x)
      elif l == "m":
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
      else:
        raise NotImplementedError(f'Layer {l} ')

    # (_b, w, h, _c) = x.shape
    # y = nn.avg_pool(x, window_shape=(w, h))
    
    # Average pool
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.classifier_width)(x)
    
    if self.tracker == True:
      assert not self.repaired
      out = nn.BatchNorm()(x, use_running_average=deterministic)
    elif self.repaired == True:
      x = nn.BatchNorm()(x, use_running_average=deterministic)
    if output_every_layer: 
      outs += [x]
    x = nn.relu(x)
    
    x = nn.Dense(self.classifier_width)(x)
    if self.tracker == True:
      assert not self.repaired
      out = nn.BatchNorm()(x, use_running_average=deterministic)
    elif self.repaired == True:
      x = nn.BatchNorm()(x, use_running_average=deterministic)
    if output_every_layer: 
      outs += [x]
    x = nn.relu(x)
    if output_every_layer:
      return (x, outs)
    return x


class MultiheadClassifier(nn.Module):
  """Classifier Layer."""
  num_classes : Sequence[int]
  logit_scale : Any
  dtype: Any = jnp.float32
  
  @nn.compact
  def __call__(self, x):
    logits = []
    for classes in self.num_classes:
      logits += [jnp.exp(self.logit_scale) * jnp.asarray(nn.Dense(classes, dtype=self.dtype, use_bias=False)(x), self.dtype)]
    return logits

class Classifier(nn.Module):
  """Classifier Layer."""
  num_classes : int
  logit_scale : Any
  dtype: Any = jnp.float32
  
  @nn.compact
  def __call__(self, x):
    logits = jnp.exp(self.logit_scale) * jnp.asarray(
      nn.Dense(self.num_classes, dtype=self.dtype, use_bias=False)(x), 
      self.dtype
      )
    return logits


class VGG(nn.Module):
  """VGG."""
  backbone_layers: any 
  classifier_width: int
  classifier: ModuleDef
  projection_dim: int
  width_multiplier : int = 1
  logit_scale_init_value: float = 2.6592 # from openai/clip-vit-base-patch32
  norm: ModuleDef = nn.LayerNorm
  dtype: Any = jnp.float32
  repaired: bool = False
  tracker: bool = False
  
  @nn.compact
  def __call__(self, x, deterministic=True, output_every_layer=False): 
    encoder_outputs = VGGEncoder(backbone_layers=self.backbone_layers, 
                          width_multiplier=self.width_multiplier,
                          classifier_width=self.classifier_width,
                          norm=self.norm, 
                          name='encoder', 
                          repaired=self.repaired, 
                          tracker=self.tracker)(x, deterministic=deterministic, output_every_layer=output_every_layer)
    
    if output_every_layer:
      outs = encoder_outputs[1]
      print(len(outs))
      encoder_outputs = encoder_outputs[0]
    
    image_features = nn.Dense(
        self.projection_dim,
        dtype=self.dtype,
        kernel_init=jax.nn.initializers.normal(0.02),
        use_bias=False,
        name='visual_projection'
    )(encoder_outputs)
    image_features /= jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    
    if self.tracker == True: 
      out = nn.BatchNorm()(image_features, use_running_average=deterministic)
    elif self.repaired == True:
      image_features = nn.BatchNorm()(image_features, use_running_average=deterministic)
    
    if output_every_layer:
      outs += [image_features ]
    
    logits = self.classifier(logit_scale=self.param("logit_scale", lambda _, shape: jnp.ones(shape) * self.logit_scale_init_value, []),
                            dtype=self.dtype, 
                            name='classifier'
                            )(image_features) 
    if output_every_layer:
      return logits, outs
    return logits
  
  def permutation_spec(self, skip_classifier=False):
    num_conv_layers = 0
    for l in self.backbone_layers:
      if isinstance(l, int):
        num_conv_layers += 1
    
    p_in = None
    p_out = 'P/encoder/Conv_0'
    axes_to_perm = {
      **conv_no_bias_axes_to_perm("encoder/Conv_0", p_in, p_out),
      **norm_axes_to_perm("encoder/LayerNorm_0", p_out)
      }
    for i in range(1,num_conv_layers):
      p_in = p_out
      p_out = f'P/encoder/Conv_{i}'
      axes_to_perm.update(**conv_no_bias_axes_to_perm(f"encoder/Conv_{i}", p_in, p_out))
      axes_to_perm.update(**norm_axes_to_perm(f"encoder/LayerNorm_{i}", p_out))
    
    
    for i in range(2):
      p_in = p_out
      p_out = f'P/encoder/Dense_{i}'
      axes_to_perm.update(**dense_axes_to_perm(f"encoder/Dense_{i}", p_in, p_out))

    p_in = p_out
    p_out = None if skip_classifier else f'P/visual_projection'
    axes_to_perm.update(**dense_no_bias_axes_to_perm(f"visual_projection", p_in, p_out))
    
    axes_to_perm.update({"logit_scale": (None,)})
    
    try:
      nheads = len(self.classifier.keywords['num_classes'])
    except:
      nheads = 1
    
    for i in range(nheads):
      p_in = p_out
      axes_to_perm.update(dense_no_bias_axes_to_perm(f"classifier/Dense_{i}", p_in, None))
    
    return permutation_spec_from_axes_to_perm(axes_to_perm)

  def initialization(self, rng, batch_shape):
    @jax.jit
    def init(*args):
      return self.init(*args)
    variables = init(rng, jnp.ones(batch_shape))
    return variables
    
  def create_train_state(self, init, tx):
    return TrainState.create(
      apply_fn=self.apply, 
      params=init['params'], 
      tx=tx, 
      )  

  def load_pretrained_params(self, init, params, **kwargs):
    return { 'params' : {
        'encoder': params['params']['encoder'], 
        'visual_projection': params['params']['visual_projection'],
        'logit_scale': params['params']['logit_scale'],
        'classifier': init['params']['classifier'], 
        }
      }
  def has_batch_norm(self):
    return False

ARCHS = {
    "vgg16": {
      'backbone_layers': [64, 64, "m", 128, 128, "m", 256, 256, 256, "m", 512, 512, 512, "m", 512, 512, 512, "m"], 
      'classifier_width': 4096
    },
    "vgg19": {
      'backbone_layers': [64, 64, "m", 128, 128, "m", 256, 256, 256, 256, "m", 512, 512, 512, 512, "m", 512, 512, 512, 512, "m"], 
      'classifier_width': 4096
    },
    "vgg16-thin-classifier": {
      'backbone_layers': [64, 64, "m", 128, 128, "m", 256, 256, 256, "m", 512, 512, 512, "m", 512, 512, 512, "m"], 
      'classifier_width': 512
    },
    "vgg16-wide": {
      'backbone_layers': [512, 512 , "m", 512, 512, "m", 512, 512, 512, "m", 512, 512, 512, "m", 512, 512, 512, "m"], 
      'classifier_width': 4096
    }
}

def get_vgg(*, num_classes, backbone_layers, classifier_width, width_multiplier, projection_dim, tracker=False, repaired=False, **kwargs):
  try:
    nheads = len(num_classes)
    classifier = partial(MultiheadClassifier, num_classes=num_classes)
  except:
    classifier = partial(Classifier, num_classes=num_classes)
  return VGG(classifier=classifier, 
             backbone_layers=backbone_layers, 
             classifier_width=classifier_width, 
             width_multiplier=width_multiplier, 
             projection_dim=projection_dim, 
             tracker=tracker, 
             repaired=repaired)


VGG16 = partial(get_vgg, backbone_layers=[64, 64, "m", 128, 128, "m", 256, 256, 256, "m", 512, 512, 512, "m", 512, 512, 512, "m"],
                classifier_width=4096)

VGG19 = partial(get_vgg, backbone_layers=[64, 64, "m", 128, 128, "m", 256, 256, 256, 256, "m", 512, 512, 512, 512, "m", 512, 512, 512, 512, "m"],
                classifier_width=4096)


  
def vgg_permutation_spec(num_classes, include_) -> PermutationSpec:
  
  return permutation_spec_from_axes_to_perm(
    {
      "encoder/Conv_0/kernel": (None, None, None, "P_Conv_0"),
      **{f"encoder/Conv_{i}/kernel": (None, None, f"P_Conv_{i-1}", f"P_Conv_{i}")
         for i in range(1, 13)},
      **{f"encoder/LayerNorm_{i}/scale": (f"P_Conv_{i}", )
         for i in range(13)},
      **{f"encoder/LayerNorm_{i}/bias": (f"P_Conv_{i}", )
         for i in range(13)},
      "encoder/Dense_0/kernel": ("P_Conv_12", "P_Dense_0"),
      "encoder/Dense_0/bias": ("P_Dense_0", ),
      "encoder/Dense_1/kernel": ("P_Dense_0", "P_Dense_1"),
      "encoder/Dense_1/bias": ("P_Dense_1", ),
      "visual_projection/kernel" : ("P_visual_projection", None), 
  })

def test():
  model = VGG16(num_classes=[3, 2, 5], projection_dim=512, width_multiplier=2)
  batch_x = jnp.ones((1, 120, 120, 3))  # (N, H, W, C) format
  rng = random.PRNGKey(0)
  params = model.init(rng, batch_x)['params']
  logits = model.apply({'params':params}, batch_x)
  print(logits)

if __name__ == "__main__":
  test()