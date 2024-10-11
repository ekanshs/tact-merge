import jax.numpy as jnp
from jax import random
from flax import traverse_util
from flax.core import freeze, unfreeze

from permutations.permutation_spec import PermutationSpec

import numpy as np
from scipy.optimize import linear_sum_assignment


conv_axes_to_perm = lambda name, p_in, p_out: {f"{name}/kernel": (None, None, p_in, p_out), 
                                               f"{name}/bias": (p_out, ), }
conv_no_bias_axes_to_perm = lambda name, p_in, p_out: {f"{name}/kernel": (None, None, p_in, p_out)}
norm_axes_to_perm = lambda name, p: {f"{name}/scale": (p, ), 
                                     f"{name}/bias": (p, )}
dense_axes_to_perm = lambda name, p_in, p_out: {f"{name}/kernel": (p_in, p_out), 
                                                f"{name}/bias": (p_out, )}
dense_no_bias_axes_to_perm = lambda name, p_in, p_out: {f"{name}/kernel": (p_in, p_out)}

def get_permuted_param(ps: PermutationSpec, perm, k: str, flat_params, except_axis=None):
  """Get parameter `k` from `params`, with the permutations applied."""
  w = flat_params[k]
  for axis, p in enumerate(ps.axes_to_perm[k]):
    # Skip the axis we're trying to permute.
    if axis == except_axis:
      continue

    # None indicates that there is no permutation relevant to that axis.
    if p is not None:
      w = jnp.take(w, perm[p], axis=axis)

  return w

def apply_permutation(ps: PermutationSpec, perm, params):
  """
  The function `apply_permutation` applies a permutation to parameters.
  """
  flat_params = flatten_params(params)
  permuted_flat_params = {k: get_permuted_param(ps, perm, k, flat_params) for k in flat_params.keys()}
  return unflatten_params(permuted_flat_params)


def _random_multi_head_embedding(rng, n, nheads, hdim):
  assert n == nheads * hdim
  rngs = random.split(rng, nheads+1)
  head_p = random.permutation(rngs[0], jnp.arange(nheads))
  rperm = []
  for ix in range(nheads):
    rperm += [random.permutation(rngs[ix+1], jnp.arange(hdim)) + hdim * head_p[ix]]
  return jnp.concatenate(rperm, axis=0)


def random_permutation(rng, ps: PermutationSpec, params):
  """Find a permutation of `params_b` to make them match `params_a`."""
  flat_params = flatten_params(params)
  perm_sizes = {p: flat_params[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
  rngs = random.split(rng, len(perm_sizes))
  perm = {p: 
          (_random_multi_head_embedding(rng, n, ps.perm_to_multi_head_embedding[p]['nheads'], ps.perm_to_multi_head_embedding[p]['hdim']) 
           if p in ps.perm_to_multi_head_embedding.keys() else random.permutation(rngs[i], n))  
          for i, (p, n) in enumerate(perm_sizes.items())}
  return perm

def identity_permutation(ps: PermutationSpec, params):
  """Find a permutation of `params_b` to make them match `params_a`."""
  flat_params = flatten_params(params)
  perm_sizes = {p: flat_params[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
  perm = {p: jnp.arange(n) for p, n in perm_sizes.items()}
  return perm

def flatten_params(params):
  return {"/".join(k): v for k, v in traverse_util.flatten_dict(unfreeze(params)).items()}

def unflatten_params(flat_params):
  return freeze(
      traverse_util.unflatten_dict({tuple(k.split("/")): v
                                    for k, v in flat_params.items()}))

def _invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asarray(p) 
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def invert_permutation(perm):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    perm = {}
    for pname, p in perm:
      perm[pname] = _invert_permutation(p)
    return perm

def find_permutation(correlation, nheads=None, hdim=None):
  if nheads is None: 
    ri, ci = linear_sum_assignment(correlation, maximize=True)    
    assert (ri == jnp.arange(len(ri))).all()
    perm = ci
  else:
    assert hdim is not None
    assert correlation.shape[0] == correlation.shape[1] == nheads * hdim
    ri, head_perm = linear_sum_assignment(jnp.einsum("ijkl->ik", correlation.reshape(nheads, hdim, nheads, hdim)), maximize=True)
    per_head_perm = []
    for i, p_i in zip(jnp.arange(nheads), head_perm):
      ri, ci = linear_sum_assignment(correlation.reshape(nheads, hdim, nheads, hdim)[i, :, p_i, :], maximize=True)
      per_head_perm += [ci + p_i * hdim]
    perm = jnp.concatenate(per_head_perm, axis=0)
  return perm

