from configs import common
from configs.common import TRAINING_SCHEDULE
import ml_collections

def get_config(config_str):
  """Returns default parameters for training/finetuning a `model` on `dataset`."""
  model, datasets, model_merge = config_str.split(',')
  datasets = datasets.split('.')
  config = ml_collections.ConfigDict(common.get_config().to_dict())
  config.model = model
  config.datasets = datasets
  config.merging_method = MODEL_MERGING[model_merge]
  config.training_schedule = TRAINING_SCHEDULE["cosine"]  
  
  for dataset in common.DATASET_PRESETS.keys():
    config[dataset] = common.DATASET_PRESETS[dataset]
  
  
  return config



MODEL_MERGING = {
  'zero-shot': ml_collections.ConfigDict({
    'name': 'zero-shot',
  }),
  'task-arithmetic': ml_collections.ConfigDict({
    'name': 'task-arithmetic',
    'min': 0.0,
    'max' : 1.0,
    'n': 10
  }),
  'average-merging': ml_collections.ConfigDict({
    'name': 'average-merging', 
    'min': 0.5,
    'max' : 1.5,
    'n': 10
  }),
  'ties-merging': ml_collections.ConfigDict({
    'name': 'ties-merging',
    'min': 0.0,
    'max' : 1.0,
    'n': 10, 
  }),}
