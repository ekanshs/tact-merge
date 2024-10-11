import jax.numpy as jnp
from jax.tree_util import tree_map

from pruning import keep_top_k
import utils

def compute_ties_vector(task_vectors, top_k=1.0):
    #TRIM
    task_vectors = [keep_top_k(vector, top_k) for vector in task_vectors]

    #SIGN-ELECT: 
    sign_vector = utils.tree_sign(utils.add_trees(task_vectors))

    #MERGE
    elected_sum_vector = utils.add_trees([tree_map(lambda v, sign: jnp.where(v*sign >= 0, v, jnp.zeros(v.shape)), t, sign_vector) for t in task_vectors])
    elected_count_vector = utils.add_trees([tree_map(lambda v, sign: jnp.where(v*sign >= 0, jnp.ones(v.shape), jnp.zeros(v.shape)), t, sign_vector) for t in task_vectors])
    ties_vector = utils.tree_divide(elected_sum_vector, elected_count_vector)
  
    return ties_vector

def compute_task_arithmetic_vector(task_vectors, lam=0.4):
    return utils.add_trees(task_vectors, lam * jnp.ones((len(task_vectors,))))

def compute_tall_mask(task_vector, task_MTL, lam=1.0):
    return tree_map(lambda t, t_mtl: jnp.where(jnp.abs(t) >= lam * jnp.abs(t_mtl - t), jnp.ones(t.shape), jnp.zeros(t.shape)), task_vector, task_MTL)

