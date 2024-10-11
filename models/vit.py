from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp
import jax

from transformers import FlaxViTModel, AutoConfig, FlaxViTForImageClassification

from tqdm import tqdm

from models.train_state import TrainState
from permutations import PermutationSpec, dense_axes_to_perm, conv_axes_to_perm, norm_axes_to_perm, permutation_spec_from_axes_to_perm, flatten_params, unflatten_params, unfreeze

from models.repaired_flax_vit import RepairedFlaxViTModule
ModuleDef=Any

# vit_module = FlaxViTModel.module_class

class ViTModelwithClassifier(nn.Module):
  config: Any
  classifier_module : ModuleDef
  dtype: jnp.dtype = jnp.float32 
  tracker: bool = False
  repaired: bool = False
  
  @nn.compact
  def __call__(
    self, 
    pixel_values, 
    deterministic: bool = True,
  ):
    
    outputs = RepairedFlaxViTModule(config=self.config, dtype=self.dtype, add_pooling_layer=False, name='encoder', tracker=self.tracker, repaired=self.repaired)(pixel_values=pixel_values, deterministic=deterministic)
    hidden_states = outputs[0]
    logits = self.classifier_module(dtype=self.dtype, 
                                    name='classifier', 
                                    kernel_init=jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
                          )(hidden_states[:, 0, :]) 
    return logits
  
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
    if params.get('params', None) is not None:
      return {
        'params': {
          'encoder': params['params']['encoder'],
          'classifier': init['params']['classifier'],  
        }
      }
    else:
      return { 'params' : {
        'encoder': params['vit'], 
        'classifier': init['params']['classifier'], 
        }
      }
  
  # def load_params_with_batch_stats(self, init, params):
  #   flat_params = flatten_params(params)
  #   flat_init = flatten_params(init)
  #   for k, v in flat_params.items():
  #     flat_init[k] = v
  #   return unfreeze(unflatten_params(flat_init))
  
  # def set_batch_norm_params(self, params, batch_stats):
  #   flat_params = flatten_params(params)
  #   flat_batch_stats = flatten_params(batch_stats)
  #   for k, v in flat_batch_stats.items():
  #     if k.split('/')[-1] == 'mean':
  #       flat_params['params/' + k.removesuffix('mean') + 'bias'] = v
  #     elif k.split('/')[-1] == 'var':
  #       flat_params['params/' + k.removesuffix('var') + 'scale'] = jnp.sqrt(v)
  #   return unflatten_params(flat_params)  
  def has_batch_norm(self):
    return False
  
  def permutation_spec(self, skip_classifier=False):    
    ## Embedder layer:
    perm_to_multi_head_embedding = {}
    embedder_layer_prefix= "encoder/embeddings"
    p_in = None
    p_out = f'P/encoder/embeddings' 
    
    perm_to_act_layers = {'P/encoder/embeddings': ['encoder/embeddings']}
    axes_to_perm = conv_axes_to_perm(f'{embedder_layer_prefix}/patch_embeddings/projection', p_in, p_out)
    axes_to_perm.update({f'{embedder_layer_prefix}/position_embeddings':  (None, None, p_out) })
    axes_to_perm.update({f'{embedder_layer_prefix}/cls_token':  (None, None, p_out) })

    ## Encoder layer:
    attention_axes_to_perm = lambda name, p: {
      **dense_axes_to_perm(f"{name}/attention/key", p, f"P/{name}/attention"),
      **dense_axes_to_perm(f"{name}/attention/value", p, f"P/{name}/attention"),
      **dense_axes_to_perm(f"{name}/attention/query", p, f"P/{name}/attention"),
      **dense_axes_to_perm(f"{name}/output/dense", f"P/{name}/attention", p),
    }

    layer_axes_to_perm = lambda name, p,: {
      **norm_axes_to_perm(f"{name}/layernorm_before", p),
      **attention_axes_to_perm(f"{name}/attention", p), 
      **norm_axes_to_perm(f"{name}/layernorm_after", p), 
      **dense_axes_to_perm(f"{name}/intermediate/dense", p, f"P/{name}/intermediate"), 
      **dense_axes_to_perm(f"{name}/output/dense", f"P/{name}/intermediate", p),   
    }

    layer_perm_to_act_layers = lambda name: [f'{name}/layernorm_before', 
                                            f"{name}/attention", 
                                            f"{name}/layernorm_after", 
                                            f"{name}/output/dense",]    
    encoder_layer_prefix = "encoder/encoder/layer"
    p = p_out
    for layer in range(self.config.num_hidden_layers):
      layer_name = f'{encoder_layer_prefix}/{layer}'
      axes_to_perm.update(layer_axes_to_perm(layer_name, p))
      perm_to_multi_head_embedding.update({
        f'P/{layer_name}/attention/attention': {
          'nheads' : self.config.num_attention_heads,
          'hdim': self.config.hidden_size // self.config.num_attention_heads,
          },
        })
      perm_to_act_layers['P/encoder/embeddings'] += layer_perm_to_act_layers(layer_name)
    axes_to_perm.update(norm_axes_to_perm(f"encoder/layernorm", p))
    perm_to_act_layers['P/encoder/embeddings'] += ["encoder/layernorm"]

    try:
      nheads = len(self.classifier_module.keywords['num_classes'])
    except:
      nheads = 1
    
    for i in range(nheads):
      p_in = p_out
      axes_to_perm.update(dense_axes_to_perm(f"classifier/Dense_{i}", p_in, None))
    
    # return permutation_spec_from_axes_to_perm(axes_to_perm, perm_to_multi_head_embedding, perm_to_act_layers=perm_to_act_layers)
    return permutation_spec_from_axes_to_perm(axes_to_perm, perm_to_multi_head_embedding, perm_to_act_layers={})



class MultiheadClassifier(nn.Module):
  """Classifier Layer."""
  num_classes : Sequence[int]
  kernel_init : Any = jax.nn.initializers.lecun_normal()
  logit_scale: Any = jnp.array(0.)
  dtype: Any = jnp.float32
  use_bias : bool = True
  
  @nn.compact
  def __call__(self, hidden_states):
    logits = []
    for classes in self.num_classes:
      logits += [
        jnp.exp(self.logit_scale) * \
          jnp.asarray(
            nn.Dense(
              classes, 
              kernel_init=self.kernel_init, 
              dtype=self.dtype, 
              use_bias=self.use_bias
              )(hidden_states), 
            self.dtype
            )
          ]
    return logits

class Classifier(nn.Module):
  """Classifier Layer."""
  num_classes : int
  kernel_init : Any = jax.nn.initializers.lecun_normal()
  logit_scale: Any = jnp.array(0.)
  dtype: Any = jnp.float32
  
  @nn.compact
  def __call__(self, hidden_states):
    logits = jnp.exp(self.logit_scale) * \
      jnp.asarray(
        nn.Dense(
          self.num_classes, 
          kernel_init=self.kernel_init,
          dtype=self.dtype, 
          use_bias=False
          )(hidden_states), 
        self.dtype
        )
    return logits

def get_vit_model_with_classifier(config, dtype, num_classes, tracker, repaired):
  try:
    nheads = len(num_classes)
    classifier = partial(MultiheadClassifier, num_classes=num_classes)
  except:
    classifier = partial(Classifier, num_classes=num_classes)
  return ViTModelwithClassifier(config=config,
                                classifier_module=classifier,
                                dtype=dtype, 
                                tracker=tracker, 
                                repaired=repaired)


def ViTB16(*, num_classes, dtype, tracker=False, repaired=False, **kwargs):
  model_config = AutoConfig.from_pretrained('google/vit-base-patch16-224')
  return get_vit_model_with_classifier(
    config=model_config,
    num_classes=num_classes, 
    dtype=dtype, 
    tracker=tracker, 
    repaired=repaired
    )


def PermViTB16(*, num_classes, dtype, num_attention_heads=-1, width_multiplier=1, depth_multiplier=1,  tracker=False, repaired=False, **kwargs):
  model_config = AutoConfig.from_pretrained('google/vit-base-patch16-224')
  model_config.intermediate_size = int(model_config.intermediate_size * width_multiplier)
  model_config.hidden_size = int(model_config.hidden_size * width_multiplier)
  model_config.num_hidden_layers = int(model_config.num_hidden_layers * depth_multiplier)
  if num_attention_heads > 0:
    model_config.num_attention_heads = num_attention_heads
  return get_vit_model_with_classifier(
    config=model_config,
    num_classes=num_classes, 
    dtype=dtype, 
    tracker=tracker, 
    repaired=repaired
    )


def ViTL14(*, num_classes, dtype, **kwargs):
  model_config = AutoConfig.from_pretrained('google/vit-large-patch14-224')
  return get_vit_model_with_classifier(
    config=model_config,
    num_classes=num_classes, 
    dtype=dtype)

def ViTB16IN21k(*, num_classes, dtype, **kwargs):
  model_config = AutoConfig.from_pretrained('google/vit-base-patch16-224-in21k')
  return get_vit_model_with_classifier(
    config=model_config,
    num_classes=num_classes, 
    dtype=dtype)


def ViTL14IN21k(*, num_classes, dtype, **kwargs):
  model_config = AutoConfig.from_pretrained('google/vit-large-patch14-224-in21k')
  return get_vit_model_with_classifier(
    config=model_config,
    num_classes=num_classes, 
    dtype=dtype)


def dinoViTB16(*, num_classes, dtype, **kwargs):
  model_config = AutoConfig.from_pretrained('facebook/dino-vitb16')
  return get_vit_model_with_classifier(
    config=model_config,
    num_classes=num_classes, 
    dtype=dtype)


def ViTmaeB16(*, num_classes, dtype, tracker=False, repaired=False, **kwargs):
  model_config = AutoConfig.from_pretrained('facebook/vit-mae-base')
  return get_vit_model_with_classifier(
    config=model_config,
    num_classes=num_classes, 
    dtype=dtype,
    tracker=tracker, 
    repaired=repaired
    )

def ViTmaeL14(*, num_classes, dtype, **kwargs):
  model_config = AutoConfig.from_pretrained('facebook/vit-mae-large')
  return get_vit_model_with_classifier(
    config=model_config,
    num_classes=num_classes, 
    dtype=dtype)


def test():
  pass

if __name__ == "__main__":
  test()
