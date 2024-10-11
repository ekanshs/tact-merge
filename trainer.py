
from absl import logging

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from flax import jax_utils

from tqdm import tqdm
import optax
from utils import tree_norm, tree_inner_prod

from optax_utils import create_learning_rate_fn, create_path_aware_tx
from data import input_pipeline

def zero_one_loss(logits, labels):
  accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
  return 1-accuracy

def cross_entropy_loss(logits, labels):
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = 1 - zero_one_loss(logits, labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def _update_per_batch_batch_stats(apply_fn, params, batch):
  variables = {'params': params['params'], 'batch_stats': params['batch_stats']}
  _, updates = apply_fn(variables, batch['image'], deterministic=False, mutable='batch_stats')
  batch_stats = updates['batch_stats']
  is_fin = jnp.array(True)
  for b in jax.tree_util.tree_leaves(updates['batch_stats']):
    is_fin &= jnp.all(lax.is_finite(b))
  
  return {
    'params': params['params'],
    'batch_stats': jax.tree_util.tree_map(
        partial(jnp.where, is_fin),
        batch_stats,
        params['batch_stats'],)
  }

def update_batch_stats(apply_fn, params, dataset, nbatches=None, prefetch=10, axis_name=None):
  if axis_name is None:
    update_per_batch_batch_stats = jax.jit(_update_per_batch_batch_stats, static_argnums=(0,))
  else:
    update_per_batch_batch_stats = jax.pmap(_update_per_batch_batch_stats, static_broadcasted_argnums=(0,))

  data_iter = input_pipeline.prefetch(dataset, prefetch, axis_name)
  ix = 0

  for batch in data_iter:
    params = update_per_batch_batch_stats(apply_fn, params, batch)
    ix+=1
    if nbatches is not None:
      if ix >= nbatches:
        break
  
  return params

  

def train_step(state, batch, has_batch_norm = False, axis_name=None):
  """Perform a single training step."""
  def loss_fn(params):
    """loss function used for training."""
    if has_batch_norm:
      logits, updates = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        batch['image'], deterministic=False, mutable='batch_stats')
      batch_stats = updates['batch_stats']
      loss = cross_entropy_loss(logits, batch['label'])
      return loss, (logits, batch_stats)
    
    logits = state.apply_fn(
        {'params': params},
        batch['image'], deterministic=False)
  
    loss = cross_entropy_loss(logits, batch['label'])
    return loss, (logits, )

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, outs), grads = grad_fn(state.params)
  logits = outs[0]
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  if axis_name is not None:
    grads = lax.pmean(grads, axis_name=axis_name)
  
  is_fin = jnp.array(True)
  for g in jax.tree_util.tree_leaves(grads):
    is_fin &= jnp.all(lax.is_finite(g))

  metrics = compute_metrics(logits, batch['label'])
  if axis_name is not None:
    metrics = lax.pmean(metrics, axis_name=axis_name)
  
  new_state = state.apply_gradients(
      grads=grads
  )
  new_state = new_state.replace(
    opt_state=jax.tree_util.tree_map(
        partial(jnp.where, is_fin),
        new_state.opt_state,
        state.opt_state,
    ),
    params=jax.tree_util.tree_map(
      partial(jnp.where, is_fin), 
      new_state.params, 
      state.params
      ),
    )
  
  if has_batch_norm:
    new_state = new_state.replace(batch_stats=jax.tree_util.tree_map(
      partial(jnp.where, is_fin), 
      outs[1], 
      state.batch_stats
      ),
    )
  metrics.update({'g_norm': tree_norm(grads)})
  return new_state, metrics


def eval_step(state, batch, has_batch_norm=False, axis_name=None):
  variables = {'params': state.params, 'batch_stats': state.batch_stats} if has_batch_norm else {'params': state.params}
  logits = state.apply_fn(variables, batch['image'], deterministic=True)
  metrics = compute_metrics(logits, batch['label'])
  if axis_name is not None:
    metrics = lax.pmean(metrics, axis_name=axis_name)
  return metrics


def linear_probe(model, params, dataset, optimizer_config, schedule_config, prefetch=10, axis_name = None):
  """
  This function is used to train a classifier with the given parameters.
  """
  
  total_train_steps = schedule_config.num_steps 
  total_warmup_steps = schedule_config.warmup_steps 
  
  if axis_name is not None:
    p_train_step = jax.pmap(train_step, static_broadcasted_argnums=(2,3, ))
  else:
    p_train_step = jax.jit(train_step, static_argnums=(2,3, ))
  
  classifier_lr_fn = create_learning_rate_fn(
    "cosine", 
    total_train_steps, total_warmup_steps, 
    learning_rate=optimizer_config.learning_rate
  )
  
  tx = create_path_aware_tx(optimizer_config, classifier_lr_fn, params['params'], ['classifier'])

  ## create classifier train state:
  state = model.create_train_state(params, tx)

  train_iter = input_pipeline.prefetch(dataset, prefetch, axis_name)
  
  if axis_name is not None:
    state = jax_utils.replicate(state)

  progress_bar = tqdm(range(1, total_train_steps + 1))
  for step in progress_bar:
    batch = next(train_iter)
    progress_bar.set_description(f"Classifier: {step} / {total_train_steps}")
    state, _ = p_train_step(state, batch, model.has_batch_norm(), axis_name)
  
  if axis_name is not None:
    state = jax_utils.unreplicate(state)

  return {'params': state.params, 'batch_stats': state.batch_stats}
