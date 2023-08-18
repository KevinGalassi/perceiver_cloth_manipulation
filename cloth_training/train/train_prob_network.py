import os, time
from tqdm import tqdm

import torch

from cloth_training.model.ProbNetwork import ProbNetwork

from cloth_training.train.utils import iterate_hyperparameters, load_dictionary
import numpy as np
from cloth_training.model.model_architecture.model_utils import set_seed

from cloth_training.train.utils import trainer

from cloth_training.model.model_architecture.dataset_gen import GymClothDataset



save_result_to = 'prob_network'
dataset_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
no_dataset = -1         # -1 to use last dataset of the folder


# Example dictionary of hyperparameters
hyperparameters = {
      'learning_rate': [1e-2, 1e-3, 1e-4],
      'batch_size': [32, 64, 128],
      'num_epochs': [150],
      'hidden_dim': [64, 128, 256],
      'val_ratio': 0.10,
      'test_ratio' : 0.03,  
}



for hparams in tqdm(iterate_hyperparameters(hyperparameters), desc='Hyperparameter Training'):
  
   # Set seed
   if 'seeds' in hparams and hparams['seeds'] is not None:
      set_seed(hparams['seeds'])
      seed = hparams['seeds']
   else :
      seed = None

   dataset = torch.load('/home/kgalassi/code/cloth_training/dataset/pull/pulldataset.pt')
   dataset.set_obs_type('pts')
   dataset.set_output_type('prob')
   dataset.to_device(torch.device('cuda'))

   num_final_test_sample = int(len(dataset) * hyperparameters['test_ratio'])
   train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_final_test_sample, num_final_test_sample])

   run_id = str(time.strftime("%m-%d-%H-%M"))


   agent = ProbNetwork(no_points=625,
                        action_dim = 2,
                        hidden_dim = hparams['hidden_dim'],
                        lr = hparams['learning_rate'],
                        seed = seed)

 
   # TRAIN
   trainer(save_result_to, hparams, agent, train_dataset, save_model=True)
