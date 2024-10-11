"""Online-ish Pearson correlation of all n x n variable pairs simultaneously."""

from typing import NamedTuple

import jax.numpy as jnp

class OnlineMean(NamedTuple):
  sum: jnp.ndarray
  count: int

  @staticmethod
  def init(num_features: int):
    return OnlineMean(sum=jnp.zeros(num_features), count=0)

  def update(self, batch: jnp.ndarray):
    return OnlineMean(self.sum + jnp.sum(batch, axis=0), self.count + batch.shape[0])

  def mean(self):
    return self.sum / self.count

class OnlineCovariance(NamedTuple):
  a_mean: jnp.ndarray  # (d, )
  b_mean: jnp.ndarray  # (d, )
  cov: jnp.ndarray  # (d, d)
  var_a: jnp.ndarray  # (d, )
  var_b: jnp.ndarray  # (d, )
  count: int

  @staticmethod
  def init(a_mean: jnp.ndarray, b_mean: jnp.ndarray):
    assert a_mean.shape == b_mean.shape
    assert len(a_mean.shape) == 1
    d = a_mean.shape[0]
    return OnlineCovariance(a_mean,
                            b_mean,
                            cov=jnp.zeros((d, d)),
                            var_a=jnp.zeros((d, )),
                            var_b=jnp.zeros((d, )),
                            count=0)

  def update(self, a_batch, b_batch):
    assert a_batch.shape == b_batch.shape
    batch_size, _ = a_batch.shape
    a_res = a_batch - self.a_mean
    b_res = b_batch - self.b_mean
    return OnlineCovariance(a_mean=self.a_mean,
                            b_mean=self.b_mean,
                            cov=self.cov + a_res.T @ b_res,
                            var_a=self.var_a + jnp.sum(a_res**2, axis=0),
                            var_b=self.var_b + jnp.sum(b_res**2, axis=0),
                            count=self.count + batch_size)

  def covariance(self):
    return self.cov / (self.count - 1)

  def a_variance(self):
    return self.var_a / (self.count - 1)

  def b_variance(self):
    return self.var_b / (self.count - 1)

  def a_stddev(self):
    return jnp.sqrt(self.a_variance())

  def b_stddev(self):
    return jnp.sqrt(self.b_variance())

  def E_ab(self):
    return self.covariance() + jnp.outer(self.a_mean, self.b_mean)

  def pearson_correlation(self):
    # Note that the 1/(n-1) normalization terms cancel out nicely here.
    # TODO: clip?
    eps = 0
    # Dead units will have zero variance, which produces NaNs. Convert those to
    # zeros with nan_to_num.
    return jnp.nan_to_num(self.cov / (jnp.sqrt(self.var_a[:, jnp.newaxis]) + eps) /
                          (jnp.sqrt(self.var_b) + eps))

class OnlineInnerProduct(NamedTuple):
  val: jnp.ndarray  # (d, d)

  @staticmethod
  def init(d: int):
    return OnlineInnerProduct(val=jnp.zeros((d, d)))

  def update(self, a_batch, b_batch):
    assert a_batch.shape == b_batch.shape
    return OnlineInnerProduct(val=self.val + a_batch.T @ b_batch)
