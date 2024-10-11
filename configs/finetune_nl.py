r"""Train/Finetune a model.

Example for training ResNet18 on CIFAR10:

python main.py --job_type=train --config=configs/default.py:ResNet18;cifar10;sgd --expdir=./experiments/ 

"""

from configs.common import TRAIN_OPTIMIZER_PRESETS, DATASET_PRESETS, TRAINING_SCHEDULE
from configs import common
import ml_collections

def get_config(model_dataset_optimizer):
  """Returns default parameters for training/finetuning a `model` on `dataset`."""
  model, dataset, opt, decay_schedule = model_dataset_optimizer.split(',')
  config = with_model_dataset_opt(common.get_config(), model, dataset, opt, decay_schedule)  
  return config

def with_model_dataset_opt(config: ml_collections.ConfigDict, model: str, dataset: str, opt:str, decay_schedule: str):
  config = ml_collections.ConfigDict(config.to_dict())
  config.model = model
  config.dataset = dataset
  config[dataset] = DATASET_PRESETS[dataset]
  config.training_schedule = TRAINING_SCHEDULE['piecewise-constant']
  config.optimizer = TRAIN_OPTIMIZER_PRESETS[opt]
  config.pretrained_dir = ''
  config.from_pretrained = True
  config.train_classifier_at_init = True
  return config.lock()

