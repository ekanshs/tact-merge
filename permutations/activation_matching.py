import jax
import jax.numpy as jnp

from permutations.util import flatten_params, find_permutation, identity_permutation, unfreeze, apply_permutation
from permutations.online_stats import OnlineMean, OnlineCovariance

from einops import rearrange

import flax.linen as nn
from data import input_pipeline
from tqdm import tqdm

def activation_matching(ps, model, params_a, params_b, data, nsteps = None, prefetch=10, axis_name = None):
  p_layers = ps.perm_to_axes.keys()
  # act_layer = lambda x : x.removeprefix('P/')+'/__call__'
  
  def _get_flat_intermediates(params, batch, axis_name = None):
    _, state = model.apply(
      params,
      batch['image'], 
      deterministic=True,
      capture_intermediates=True,
      mutable=['intermediates']
      )
    return flatten_params(state['intermediates'])
  
  if axis_name is None:
    get_flat_intermediates = jax.jit(_get_flat_intermediates)
  else:
    get_flat_intermediates = jax.pmap(_get_flat_intermediates)
  
  def extract_act(intermediates, layer):
    act = intermediates[layer][0]
    if type(act) == tuple:
      act = act[0]
    if act.ndim==4:
      act = rearrange(act, "batch w h c -> (batch w h) c")
    if act.ndim==3:
      act = rearrange(act, "batch w c -> (batch w) c")
    if act.ndim != 2 :
      raise AssertionError(f"Flattened Layer {layer}, activation dimension is not compatible: act.shape = {act.shape}")  
    return act

  
  # Calculate mean activations
  def _calc_means():
    def one(params):
      means = {p: OnlineMean.init(flatten_params(params['params'])[axes[0][0]].shape[axes[0][1]]) for p, axes in ps.perm_to_axes.items()}
      data_iter = input_pipeline.prefetch(data, prefetch, axis_name)
      for step, batch in tqdm(enumerate(data_iter)):
        flat_intermediates = get_flat_intermediates(params,batch)
        # means = {p_layer: means[p_layer].update(extract_act(flat_intermediates, act_layer(p_layer))) for p_layer in p_layers}
        for p_layer in p_layers:
          if ps.perm_to_act_layers.get(p_layer, None) is not None:
            for act_layer in ps.perm_to_act_layers[p_layer]:
              means[p_layer] = means[p_layer].update(extract_act(flat_intermediates, act_layer + '/__call__'))
          else:
            act_layer = p_layer.removeprefix('P/') + '/__call__'
            means[p_layer] = means[p_layer].update(extract_act(flat_intermediates, act_layer))
        if nsteps is not None:
          if step+1 >= nsteps:
            break
      return means

    return one(params_a), one(params_b)

  a_means, b_means = _calc_means()
  # Calculate the Pearson correlation between activations of the two models on
  # each layer.
  def _calc_corr():  
    stats = {
        p_layer: OnlineCovariance.init(a_means[p_layer].mean(), b_means[p_layer].mean())
        for p_layer in p_layers
    }
    data_iter = input_pipeline.prefetch(data, prefetch, axis_name)
    for step, batch in tqdm(enumerate(data_iter)):
      flat_intermediates_a = get_flat_intermediates(params_a, batch)
      flat_intermediates_b = get_flat_intermediates(params_b, batch)
      for p_layer in p_layers:
        if ps.perm_to_act_layers.get(p_layer, None) is not None:
          for act_layer in ps.perm_to_act_layers[p_layer]:
            stats[p_layer] = stats[p_layer].update(extract_act(flat_intermediates_a, act_layer + '/__call__'), 
                                  extract_act(flat_intermediates_b, act_layer + '/__call__'), )
        else:
          act_layer = p_layer.removeprefix('P/') + '/__call__'
          stats[p_layer] = stats[p_layer].update(extract_act(flat_intermediates_a, act_layer), 
                                extract_act(flat_intermediates_b, act_layer), )

      # stats = {p_layer: stats[p_layer].update(extract_act(flat_intermediates_a, act_layer(p_layer)), 
      #                                         extract_act(flat_intermediates_b, act_layer(p_layer))) 
      #         for p_layer in p_layers}
      if nsteps is not None:
        if step+1 >= nsteps:
          break
    return stats
  
  cov_stats = _calc_corr()  

  perm =  {}
  for p_layer in p_layers:
    nheads = hdim = None
    if ps.perm_to_multi_head_embedding.get(p_layer, None) is not None:
      nheads = ps.perm_to_multi_head_embedding[p_layer]['nheads']
      hdim = ps.perm_to_multi_head_embedding[p_layer]['hdim']
    perm[p_layer] = find_permutation(cov_stats[p_layer].pearson_correlation(), nheads, hdim)

  return perm

# def batch_activation_matching(ps, model, params_a, params_b, batch, axis_name = None):
#   p_layers = ps.perm_to_axes.keys()
  
#   def _get_flat_intermediates(params, batch, axis_name = None):
#     _, state = model.apply(
#       params,
#       batch['image'], 
#       deterministic=True,
#       capture_intermediates=True,
#       mutable=['intermediates']
#       )
#     return flatten_params(state['intermediates'])
  
#   if axis_name is None:
#     get_flat_intermediates = jax.jit(_get_flat_intermediates)
#   else:
#     get_flat_intermediates = jax.pmap(_get_flat_intermediates)
  
#   def extract_act(intermediates, layer):
#     act = intermediates[layer][0]
#     if type(act) == tuple:
#       act = act[0]
#     if act.ndim==4:
#       act = rearrange(act, "batch w h c -> (batch w h) c")
#     if act.ndim==3:
#       act = rearrange(act, "batch w c -> (batch w) c")
#     if act.ndim != 2 :
#       raise AssertionError(f"Flattened Layer {layer}, activation dimension is not compatible: act.shape = {act.shape}")  
#     return act

  
#   # Calculate mean activations
#   def _calc_means():
#     def one(params):
#       means = {p: OnlineMean.init(flatten_params(params['params'])[axes[0][0]].shape[axes[0][1]]) for p, axes in ps.perm_to_axes.items()}
#       flat_intermediates = get_flat_intermediates(params,batch)
#       # means = {p_layer: means[p_layer].update(extract_act(flat_intermediates, act_layer(p_layer))) for p_layer in p_layers}
#       for p_layer in p_layers:
#         if ps.perm_to_act_layers.get(p_layer, None) is not None:
#           for act_layer in ps.perm_to_act_layers[p_layer]:
#             means[p_layer] = means[p_layer].update(extract_act(flat_intermediates, act_layer + '/__call__'))
#         else:
#           act_layer = p_layer.removeprefix('P/') + '/__call__'
#           means[p_layer] = means[p_layer].update(extract_act(flat_intermediates, act_layer))
#       return means

#     return one(params_a), one(params_b)

#   a_means, b_means = _calc_means()
#   # Calculate the Pearson correlation between activations of the two models on
#   # each layer.
#   def _calc_corr():  
#     stats = {
#         p_layer: OnlineCovariance.init(a_means[p_layer].mean(), b_means[p_layer].mean())
#         for p_layer in p_layers
#     }
#     data_iter = input_pipeline.prefetch(data, prefetch, axis_name)
#     for step, batch in tqdm(enumerate(data_iter)):
#       flat_intermediates_a = get_flat_intermediates(params_a, batch)
#       flat_intermediates_b = get_flat_intermediates(params_b, batch)
#       for p_layer in p_layers:
#         if ps.perm_to_act_layers.get(p_layer, None) is not None:
#           for act_layer in ps.perm_to_act_layers[p_layer]:
#             stats[p_layer] = stats[p_layer].update(extract_act(flat_intermediates_a, act_layer + '/__call__'), 
#                                   extract_act(flat_intermediates_b, act_layer + '/__call__'), )
#         else:
#           act_layer = p_layer.removeprefix('P/') + '/__call__'
#           stats[p_layer] = stats[p_layer].update(extract_act(flat_intermediates_a, act_layer), 
#                                 extract_act(flat_intermediates_b, act_layer), )

#       # stats = {p_layer: stats[p_layer].update(extract_act(flat_intermediates_a, act_layer(p_layer)), 
#       #                                         extract_act(flat_intermediates_b, act_layer(p_layer))) 
#       #         for p_layer in p_layers}
#       if nsteps is not None:
#         if step+1 >= nsteps:
#           break
#     return stats
  
#   cov_stats = _calc_corr()  

#   perm =  {}
#   for p_layer in p_layers:
#     nheads = hdim = None
#     if ps.perm_to_multi_head_embedding.get(p_layer, None) is not None:
#       nheads = ps.perm_to_multi_head_embedding[p_layer]['nheads']
#       hdim = ps.perm_to_multi_head_embedding[p_layer]['hdim']
#     perm[p_layer] = find_permutation(cov_stats[p_layer].pearson_correlation(), nheads, hdim)



# def randomized_activation_matching(ps, model, params_a, params_b, data, nsteps = 100, prefetch=10, axis_name = None):
#   final_perm = identity_permutation(ps, params_a)
#   curr_perm = identity_permutation(ps, params_a)
#   for i in range(nsteps):
#     params_b = unfreeze(apply_permutation(ps, curr_perm, params_b))
#     curr_perm = activation_matching(ps, model, params_a, params_b, data, nsteps, prefetch, axis_name)
#     final_perm = {p: v[curr_perm[p]] for p,v in final_perm.items()}
#   return final_perm