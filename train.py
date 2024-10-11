"""
This script trains a model on a given dataset
The data is loaded using tensorflow_datasets.
"""

import os
from absl import logging

import jax
from jax import random

from flax import jax_utils
from flax.training import common_utils

import ml_collections

from data import input_pipeline
import models

import trainer

from tqdm import tqdm
import wandb

from optax_utils import create_learning_rate_fn, create_tx


from utils import save_checkpoint, restore_checkpoint

def get_metrics(metrics, axis_name):
  if axis_name is not None:
    return common_utils.get_metrics(metrics)
  else:
    return common_utils.stack_forest(metrics)


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) :
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
  ## Setup wandb
  wandb_run = wandb.init(
    project="train-vision-model",
    entity="ekanshs",
    tags=[config.dataset, config.model, f'{config.optimizer.name}_lr_{config.optimizer.learning_rate}'],
    mode="online",
    job_type='train-and-evaluate',
    config=config
  )
  
  rng = random.PRNGKey(config.seed)
  
  axis_name = None
  if jax.device_count() > 1:
    axis_name = 'batch'
    train_step = jax.pmap(trainer.train_step, axis_name=axis_name, donate_argnums=(0,), static_broadcasted_argnums=(2,3,))
    eval_step = jax.pmap(trainer.eval_step, axis_name=axis_name, static_broadcasted_argnums=(2,3,))
  else:
    train_step = jax.jit(trainer.train_step, donate_argnums=(0,), static_argnums=(2,3,))
    eval_step = jax.jit(trainer.eval_step, static_argnums=(2,3,))

  def compute_loss_and_accuracy(state, dataset, nbatches=None):
    eval_iter = input_pipeline.prefetch(dataset, config.prefetch, axis_name)
    eval_metrics = []
    ix = 0
    for eval_batch in eval_iter:
        metrics = eval_step(state, eval_batch, model.has_batch_norm(), axis_name)
        eval_metrics.append(metrics)
        ix+=1
        if nbatches is not None:
            if ix >= nbatches:
                break
    eval_metrics = get_metrics(eval_metrics, axis_name)
    summary = {
            f'eval_{k}': v
            for k, v in jax.tree_util.tree_map(
                lambda x: x.mean(), eval_metrics
            ).items()
        }
    return summary

  # Setup input pipeline
  dataset = config.dataset
  pp = config[dataset].pp
  dataset_info = input_pipeline.get_dataset_info(dataset, pp['train'])
  num_classes = dataset_info['num_classes']
  num_train_examples = dataset_info['num_examples']
  
  ds_train, ds_test = input_pipeline.get_datasets(config, dataset)
  
  logging.info(ds_train)
  logging.info(ds_test)
    
  # Setup model and train state
  model = models.create_model(
    model_cls= getattr(models, config.model), 
    num_classes=num_classes, 
    width_multiplier=config.width_multiplier,
    depth_multiplier=config.depth_multiplier,
    num_attention_heads=config.num_attention_heads,
    projection_dim=512, 
    half_precision=config.half_precision)  
    
  init_params = model.initialization(rng, batch_shape=(1,config[dataset].pp.crop, config[dataset].pp.crop, 3))
  

  def compute_loss_and_accuracy(state, dataset, nbatches=None):
    eval_iter = input_pipeline.prefetch(dataset, config.prefetch, axis_name)
    eval_metrics = []
    ix = 0
    for eval_batch in eval_iter:
        metrics = eval_step(state, eval_batch, model.has_batch_norm(), axis_name)
        eval_metrics.append(metrics)
        ix+=1
        if nbatches is not None:
            if ix >= nbatches:
                break
    eval_metrics = get_metrics(eval_metrics, axis_name)
    summary = {
            f'eval_{k}': v
            for k, v in jax.tree_util.tree_map(
                lambda x: x.mean(), eval_metrics
            ).items()
        }
    return summary

  if config.from_pretrained:
    logging.info("Finetune model from a pre-trained model.")
    logging.info(config.pretrained_dir)
    pretrained_raw_state = restore_checkpoint(config.pretrained_dir)
    if pretrained_raw_state is not None:
      init_params = model.load_pretrained_params(init_params, pretrained_raw_state)
    elif config.zero_shot_classifier:
      assert "CLIP" in config.model
      init_params = models.get_zero_shot_params(config.model, dataset=dataset)
    else:
      init_params = models.get_hf_pretrained_params(model, init_params, config.keep_batch_stats)
    if model.has_batch_norm() and not config.keep_batch_stats:
      logging.info("Updating batch_stats")
      init_params = trainer.update_batch_stats(
        apply_fn=model.apply, 
        params=init_params, 
        dataset=ds_train, 
        nbatches=100, ## Forward pass through data. This probably is an overkill. 
        prefetch=config.prefetch, 
        axis_name=axis_name)


  save_initialization = True
  if config.train_classifier_at_init:
    raw_init_state = restore_checkpoint(os.path.join(workdir, "init"))
    if raw_init_state is None:
      logging.info("Running linear probe to train the classifier layer.")
      init_params = trainer.linear_probe(
        model, init_params, ds_train, 
        config.optimizer.classifier, config.training_schedule.classifier, 
        config.prefetch, axis_name)
      logging.info(f"Classifier layer trained.") 
    else:
      logging.info("Loaded init parameters from init dir.")
      init_params = {"params": raw_init_state['params']}
      if model.has_batch_norm():
        init_params['batch_stats'] = raw_init_state['batch_stats']
      save_initialization = False


  # Training arguments
  train_batch_size = int(config.training_schedule.per_device_train_batch_size) * jax.device_count()
  steps_per_epoch = num_train_examples // train_batch_size
  num_epochs = int(config.training_schedule.num_steps/steps_per_epoch)
  total_train_steps = int(config.training_schedule.num_steps)
  total_warmup_steps = int(config.training_schedule.warmup_steps)
  
  milestones = list(config.training_schedule.get('milestones', []))
  gamma = config.training_schedule.get('gamma', 0.1)
  learning_rate_fn = create_learning_rate_fn(
        config.training_schedule.decay_schedule, 
        total_train_steps, total_warmup_steps, 
        learning_rate=config.optimizer.learning_rate, 
        milestones=milestones, 
        gamma=gamma
        )

  tx = create_tx(config.optimizer, learning_rate_fn)
  state = model.create_train_state(init_params, tx)

  if save_initialization:
    save_checkpoint(os.path.join(workdir, "init"), state, overwrite=True)

  if axis_name is not None:
    state = jax_utils.replicate(state)
  
  init_summary = compute_loss_and_accuracy(state, ds_test, nbatches=None)
  init_summary.update({"step": 0})
  logging.info(f"Test loss at initialization = {init_summary['eval_loss']}.")
  logging.info(f"Test accuracy at initialization = {init_summary['eval_accuracy']}.")
  wandb_run.log(init_summary)
  if axis_name is not None:
    state = jax_utils.unreplicate(state)
  
  state = restore_checkpoint(workdir, state)
  
  step_offset = int(state.step)
  
  if axis_name is not None:
    state = jax_utils.replicate(state)
  
  train_iter = input_pipeline.prefetch(ds_train, config.prefetch, axis_name)
  train_metrics = []
  progress_bar = tqdm(range(step_offset + 1, total_train_steps + 1))
  
  for step in progress_bar:
    epoch = step // steps_per_epoch
    progress_bar.set_description(f"Epoch: {epoch} / {num_epochs}")
    batch = next(train_iter)
    state, metrics = train_step(state, batch, model.has_batch_norm(), axis_name)
    
    train_metrics.append(metrics)
    if (step + 1) % config.progress_every == 0:
      train_metrics = get_metrics(train_metrics, axis_name)
      summary = {
            k: float(v)
            for k, v in jax.tree_util.tree_map(
                lambda x: x.mean(), train_metrics
            ).items()
        }
      summary.update({"step": step})
      summary.update({"learning_rate": learning_rate_fn(step)})
      wandb_run.log(summary)
      train_metrics = []
    
    if step == step_offset + 1:
      logging.info("Initial compilation complete")
    
    if step % config.checkpoint_every == 0 or step == total_train_steps:
      if axis_name is not None:
        save_checkpoint(workdir, jax_utils.unreplicate(state))
      else:
        save_checkpoint(workdir, state)
    
    if step % config.eval_every == 0 or step == total_train_steps:
      summary = compute_loss_and_accuracy(state, ds_test)
      summary.update({"step": step})
      wandb_run.log(summary)
      progress_bar.set_postfix_str(f"Eval loss: {summary['eval_loss']:0.4f}\t Eval accuracy: {summary['eval_accuracy'] * 100 :0.2f}%")
  
  jax.random.normal(jax.random.key(0), ()).block_until_ready()

  wandb_run.finish()
  return 

