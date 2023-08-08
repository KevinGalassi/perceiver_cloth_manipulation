

import time

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import os
import pickle


import torch


def custom_collate(batch):
   
   gt_prob_batch = []
   gt_action_batch = []

   b               = [len(batch)]
   obs_size        = list(batch[0][0].shape)
   gt_prob_size    = list(batch[0][1][0].shape)
   gt_action_size  = list(batch[0][1][1].shape)

   obs_batch       = torch.empty((b + obs_size), device='cuda')
   gt_prob_batch   = torch.empty((b + gt_prob_size), device='cuda')
   gt_action_batch = torch.empty((b + gt_action_size), device='cuda')

   for i, sample in enumerate(batch):
      obs, (gt_prob, gt_action) = sample
      obs_batch[i,:] = obs
      gt_prob_batch[i,:] = gt_prob
      gt_action_batch[i,:] = gt_action

   return obs_batch, (gt_prob_batch, gt_action_batch)

def trainer(folder_name, hparams, agent, dataset, write_log = True, save_model=False, test=False) :

   log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', folder_name)
   model_base_path   = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model', folder_name)
   run_id = time.strftime("%m-%d-%H-%M")

   print('Using params ', hparams)

   if write_log:
      from torch.utils.tensorboard import SummaryWriter
      print('Logging to ', os.path.join(log_path, str(run_id)))
      writer = SummaryWriter(log_dir=os.path.join(log_path, str(run_id)))
      writer.add_text('Hyperparameters', str(hparams))
      writer.flush()
      


   if 'test_ratio' in hparams :
      test_ratio = hparams['test_ratio']
   else :
      test_ratio = 0.05

   num_val_samples = int(len(dataset) * hparams['val_ratio'])
   num_final_test_sample = int(len(dataset) * test_ratio)
   num_train_samples = len(dataset) - num_val_samples - num_final_test_sample

   train_d, val_d, test_d = torch.utils.data.random_split(dataset, [num_train_samples, num_val_samples, num_final_test_sample])

   train_loader = DataLoader(train_d, batch_size=hparams['batch_size'], shuffle=True, collate_fn=custom_collate)
   val_loader   = DataLoader(val_d, batch_size=hparams['batch_size'], shuffle=False, collate_fn=custom_collate)
   test_loader  = DataLoader(test_d, batch_size=1, shuffle=False)


   print(f'Logging results to : {folder_name}')
   for epoch in tqdm(range(hparams['num_epochs']), desc='Epoch training'):

      epoch_train_result = agent.trainer(train_loader)
      if write_log :
         for key, value in epoch_train_result.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
         writer.flush()

      epoch_val_result = agent.validate(val_loader)

      if write_log :
         for key, value in epoch_val_result.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
         writer.flush()

      if save_model :
         model_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model', folder_name)
         model_path = os.path.join(model_base_path, str(run_id)+'.pth')
         agent.save_best_model(model_path)
         
         with open(os.path.join(model_base_path, f'{run_id}_hparams.pickle'), 'wb') as f:
            pickle.dump(hparams, f, protocol=4)
         

   if test :


      model = torch.load(model_path)

      agent = agent.my_object.__init__(**model['init_params'])
      agent.load_model(model['model'])

      test_result = agent.test(test_loader)
      if write_log :
         '''
         {'val_loss': prob_loss.item(),
                              'accuracy' : a,
                              'precision4': p4,
                              'precision3': p3,
                              'precision2': p2,
                              }
         '''

         for data_no, dict_data in test_result['data'] :
            for key, value in dict_data.items():
               writer.add_scalar(f'Test/{key}', value, data_no)
               writer.flush()       

         for key, value in test_result['log'].items():
            writer.add_scalar(f'Test/mean_{key}', value, epoch)
            writer.flush()


   
