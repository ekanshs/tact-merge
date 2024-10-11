from flax import traverse_util
import optax
from typing import Callable
import jax.numpy as jnp

import logging

def create_learning_rate_fn(
  decay_schedule:str, num_train_steps: int, num_warmup_steps: int, learning_rate: float, milestones = [], gamma = 0.1 
) -> Callable[[int], jnp.ndarray]:
  """Returns a linear warmup, with linear or cosine decay learning rate function."""  
  warmup_fn = optax.linear_schedule(
    init_value=0.0, 
    end_value=learning_rate, 
    transition_steps=num_warmup_steps
  )
  if decay_schedule == 'cosine':
    decay_fn = optax.cosine_decay_schedule(
        init_value=learning_rate, 
        decay_steps=num_train_steps - num_warmup_steps
    )
  elif decay_schedule == 'linear':
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, 
        end_value=0.,
        transition_steps=num_train_steps - num_warmup_steps
    )
  elif decay_schedule == 'piecewise-constant':
    boundaries_and_scales = {i - num_warmup_steps : gamma for i in milestones}
    decay_fn = optax.piecewise_constant_schedule(
                      init_value=learning_rate, 
                      boundaries_and_scales=boundaries_and_scales)
  else:
    logging.info("Can't find a learning rate decay schedule. Defaulting to constant learning_rate schedule")
    decay_fn = optax.constant_schedule(value=learning_rate)
    
  schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
  return schedule_fn


def create_tx(config, learning_rate_fn):
  """
  The function `create_tx` creates an optimization transformation (`tx`) based on the provided
  configuration (`config`).
  
  :param config: The `config` parameter is an object that contains various configuration options for
  creating the transaction (`tx`). It should have the following attributes:
  :return: a transformation (tx) that is created based on the configuration provided. The type of
  transformation returned depends on the value of the "name" attribute in the config. If the name is
  "sgd", the transformation is created using the optax.chain and optax.sgd functions. If the name is
  "adam", the transformation is created using the optax.adamw function
  """
  def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    # find out all LayerNorm parameters
    layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
    layer_norm_named_params = {
        layer[-2:]
        for layer_norm_name in layer_norm_candidates
        for layer in flat_params.keys()
        if layer_norm_name in "".join(layer).lower()
    }
    flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)

  if config.name == "sgd":
    # tx = optax.chain( optax.clip_by_global_norm(1.0), optax.add_decayed_weights(config.weight_decay, mask=decay_mask_fn), 
    #                   optax.sgd(learning_rate_fn, momentum=config.momentum))
    if config.clip_global_norm is not None:
      tx = optax.chain( optax.clip_by_global_norm(config.clip_global_norm), optax.add_decayed_weights(config.weight_decay, mask=decay_mask_fn), 
                      optax.sgd(learning_rate_fn, momentum=config.momentum))
    else:
      tx = optax.chain(optax.add_decayed_weights(config.weight_decay, mask=decay_mask_fn), optax.sgd(learning_rate_fn, momentum=config.momentum))
      

  elif config.name == "adam":
    # tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate_fn, b1=config.b1, b2=config.b2, eps =config.eps, weight_decay=config.weight_decay, mask=decay_mask_fn))
    if config.clip_global_norm is not None:
      tx = optax.chain(optax.clip_by_global_norm(config.clip_global_norm), optax.adamw(learning_rate_fn, b1=config.b1, b2=config.b2, eps =config.eps, weight_decay=config.weight_decay, mask=decay_mask_fn))
    else:
      tx = optax.adamw(learning_rate_fn, b1=config.b1, b2=config.b2, eps =config.eps, weight_decay=config.weight_decay)
  return tx



def create_path_aware_tx(config, learning_rate_fn, params, keywords):
  optimizer = create_tx(config, learning_rate_fn)
  partition_optimizers = {'trainable': optimizer, 'frozen': optax.set_to_zero()}
  param_partitions = traverse_util.path_aware_map(
    lambda path, v: 'trainable' if any([keyword in path for keyword in keywords]) else 'frozen', params)
  
  tx = optax.multi_transform(partition_optimizers, param_partitions)
  return tx
