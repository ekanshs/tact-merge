from absl import logging
import operator
from functools import reduce
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import traverse_util
from flax.core import freeze, unfreeze
from flax.training import checkpoints
from jax import random
from jax.tree_util import tree_reduce, tree_flatten, tree_map, tree_map_with_path


# The AttributeDict class is a subclass of the dict class that allows accessing dictionary values
# using dot notation.
class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value


def flatten_params(params):
  return {"/".join(k): v for k, v in traverse_util.flatten_dict(unfreeze(params)).items()}

def unflatten_params(flat_params):
  return traverse_util.unflatten_dict({tuple(k.split("/")): v
                                    for k, v in flat_params.items()})


## Tree Operations:  
def lerp(lam, t1, t2):
  return tree_map(lambda a, b: (1 - lam) * a + lam * b, t1, t2)

def l2_dist(t1, t2):
  return jnp.sqrt(tree_reduce(operator.add, tree_map(lambda x1, x2: jnp.sum((x1 - x2)**2), t1, t2)))

def forest_stack(trees):
    return tree_map(lambda *v: jnp.stack(v), *trees)

def forest_unstack(forest):
    leaves, treedef = tree_flatten(forest)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves)]

def tree_zeros_like(t):
  return tree_map(lambda x: jnp.zeros_like(x), t)

def tree_ones_like(t):
  return tree_map(lambda x: jnp.ones_like(x), t)

def tree_add(t1, t2):
  return tree_map(lambda x,y: x+y, t1, t2)

def tree_subtract(t1, t2):
  return tree_map(lambda x,y: x-y, t1, t2)

def tree_multiply(t1, t2):
  return tree_map(lambda l1, l2: l1*l2, t1, t2)

def tree_divide(t1, t2):
  return tree_map(lambda l1, l2: l1 / l2, t1, t2)

def tree_square(t):
  return tree_map(lambda x: jnp.square(x), t)

def tree_scalar_multiply(c, t):
  return tree_map(lambda x: c*x , t)

def tree_norm(t):
  return jnp.sqrt(tree_reduce(operator.add, tree_map(lambda x: jnp.sum(x**2), t)))

def tree_sum(t):
  return tree_reduce(operator.add, tree_map(lambda x: jnp.sum(x), t))

def normalize_tree(t):
  t_norm = tree_norm(t)
  return (tree_scalar_multiply(1/t_norm, t), t_norm)

# def add_trees(ts):
#   return tree_map(lambda *v: reduce(lambda x,y: x+y, v), *ts)

def add_trees(ts, ws=None):
  if ws is None:
      return tree_map(lambda *v: reduce(lambda x,y: x+y, v), *ts)
  ts = [tree_scalar_multiply(w, t) for w,t in zip(ws, ts)]
  return tree_map(lambda *v: reduce(lambda x,y: x+y, v), *ts)

def tree_sign(t):
  return tree_map(lambda x: jnp.sign(x) , t)

def reduce_tree_add(ts, weights=None):
  if weights is None:
    weights = jnp.ones((len(ts),) )
  return tree_map(lambda *v: weighted_sum(weights,jnp.stack(v)), *ts)


_expand_columns = lambda x, ndim: jnp.expand_dims(x, jnp.arange(1, ndim))

def column_wise_broadcast_and_add(array, weights:Sequence[float]):
  assert len(weights) == len(array)
  weights = _expand_columns(jnp.array(weights), array.ndim)
  return weights + array

def column_wise_broadcast_and_multiply(array, weights:Sequence[float]):
  assert len(weights) == len(array)
  weights = _expand_columns(jnp.array(weights), array.ndim)
  return weights * array

def weighted_sum(array, ws:Sequence[float]):
  weighted_array = column_wise_broadcast_and_multiply(array, ws)
  return jnp.sum(weighted_array, axis=0)


def expand_dims_from_tree(weights, tree):
  return tree_map(lambda x: _expand_columns(weights, x.ndim), tree)

# def forest_weighted_sum(tree_stack, weights):
#   return tree_map(lambda x , w: jnp.sum(x * w, axis=0), tree_stack, weight_stack)

def forest_stack_weighted_sum(tree_stack, weight_stack):
  return tree_map(lambda x , w: jnp.sum(x * w, axis=0), tree_stack, weight_stack)


def forest_stack_mean(tree_stack, axis=0):
  return tree_map(lambda x: x.mean(axis=axis), tree_stack)

def tree_inner_prod(t1, t2):
  return tree_sum(tree_multiply(t1, t2))

def tree_count(t):
  return sum(x.size for x in jax.tree_leaves(t))

def cosine_similarity(t1,t2):
  return tree_inner_prod(t1, t2) / (tree_norm(t1)*tree_norm(t2))



def normal_tree_like(rng, t):
  return tree_map(lambda x: random.normal(rng, shape=x.shape), t)

def rademacher_tree_like(rng, t):
  return tree_map(lambda x: random.rademacher(rng, shape=x.shape, dtype=x.dtype), t)

def projection(t1, t2):
  '''
  project t1 on t2
  '''
  return tree_scalar_multiply(tree_inner_prod(t1, t2), t2)

def orthnormal(t, ts):
    """
    make vector t orthogonal to each vector in ts.
    afterwards, normalize the output w
    """
    for t_ in ts:
        t = tree_subtract(t, projection(t, t_))
    return normalize_tree(t)[0]


def restore_checkpoint(workdir, target=None):
  return checkpoints.restore_checkpoint(workdir, target=target)

def save_checkpoint(workdir, state, keep=1, keep_every_n_steps=None, overwrite=False):
  step = int(state.step)
  logging.info('Saving checkpoint step %d.', step)
  checkpoints.save_checkpoint(workdir, state, step, keep=keep, keep_every_n_steps=keep_every_n_steps, overwrite=overwrite) 


def merge_batch_stats(expert_batch_stats_ls):
  def merge(k, *v):
    N = len(v)
    if k[-1].key == 'mean':
      return (1/N) * reduce(lambda x,y: x+y, v) 
    elif k[-1].key == 'var':
      return jnp.square((1/N) *  reduce(lambda x,y: jnp.sqrt(x)+jnp.sqrt(y), v))
  return tree_map_with_path(merge, *expert_batch_stats_ls)
