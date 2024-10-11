"""A collection of pruning utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any
import jax.numpy as jnp
from jax import tree_map
from flax import traverse_util



def apply_mask(params, mask={}):
  flat_masks = traverse_util.flatten_dict(mask)
  flat_params = traverse_util.flatten_dict(params)
  return traverse_util.unflatten_dict({k: jnp.multiply(v, flat_masks.get(k, 1)) for k, v in flat_params.items()})


def percents_init(mask, percents):
  flat_masks = traverse_util.flatten_dict(mask)
  return traverse_util.unflatten_dict({k: percents for k in flat_masks.keys()})


def prune_by_percent(percents, mask, params):
  def prune_by_percent_once(percent, mask, final_weight):
    # Put the weights that aren't masked out in sorted order.
    sorted_weights = jnp.sort(jnp.abs(final_weight[mask == 1]))

    # Determine the cutoff for weights to be pruned.
    cutoff_index = jnp.round(percent * sorted_weights.size).astype(int)
    cutoff = sorted_weights[cutoff_index]

    # Prune all weights below the cutoff.
    return jnp.where(jnp.abs(final_weight) <= cutoff, jnp.zeros(mask.shape, dtype=int), mask)

  flat_masks = traverse_util.flatten_dict(mask)
  flat_params = traverse_util.flatten_dict(params)
  flat_percents = traverse_util.flatten_dict(percents)
  return traverse_util.unflatten_dict({k: prune_by_percent_once(percent, flat_masks[k], flat_params[k]) for k, percent in flat_percents.items()})


def mask_init(params):
  masks = {}
  for k, v in traverse_util.flatten_dict(params).items():
    masks[k] = jnp.ones_like(v, dtype=int)
  return traverse_util.unflatten_dict(masks)


def keep_top_k(params, topk):
  init_mask = mask_init(params)
  percents = percents_init(init_mask, 1.-topk)
  return apply_mask(params, prune_by_percent(percents, init_mask, params))

