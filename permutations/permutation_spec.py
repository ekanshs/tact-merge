from collections import defaultdict
from typing import NamedTuple


class PermutationSpec(NamedTuple):
  perm_to_axes: dict
  axes_to_perm: dict
  perm_to_multi_head_embedding: dict
  perm_to_act_layers: dict
  
def permutation_spec_from_axes_to_perm(axes_to_perm: dict, perm_to_multi_head_embedding:dict={}, perm_to_act_layers:dict={}) -> PermutationSpec:
  perm_to_axes = defaultdict(list)
  for wk, axis_perms in axes_to_perm.items():
    for axis, perm in enumerate(axis_perms):
      if perm is not None:
        perm_to_axes[perm].append((wk, axis))
  return PermutationSpec(
    perm_to_axes=dict(perm_to_axes), 
    axes_to_perm=axes_to_perm, 
    perm_to_multi_head_embedding=perm_to_multi_head_embedding,
    perm_to_act_layers=perm_to_act_layers
    )

