import torch
from cloth_training.dataset.dataset_gen import GymClothDataset
from cloth_training.model.paper.dual_perceiver import DualPerceiver

import os, pickle, time
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from cloth_training.model.common.model_utils import set_seed
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__' :

   folder_name = 'dual_perceiver'
   save_model_path = './saved_model'
   write_log = True
   save_model = True
   dataset_path = './dataset/ablation/ablation.pt'

   #load hparams from file
   hyperparameters = [
                        {'num_epochs' : 300,
                        'batch_size' : 128,            
                        'val_ratio'  : 0.03,
                        'dataset_ratio' : 0.95,
                        'seed' : 42,
                        'input_dim' : 3,

                        #Point Network
                        'point_latent_heads' : 8,
                        'point_layers' : 3,
                        'lr_point' : 1e-4,

                        # Action parameters
                        'action_latent_heads' : 8,
                        'action_cross_heads' : 8,
                        'action_latent_layers' : 3,
                        'action_depth' : 3,
                        'lr_action' : 1e-4,
                        },
                     ]


   for hparams in hyperparameters :
      try :
         print(f'Start run with hyperparameters: \n {hparams}')

         set_seed(hparams['seed'])

         # Iterating over all combinations of hyperparameters
         dataset = torch.load(dataset_path)
         dataset.set_obs_type('heatmap')
         dataset.set_output_type('heatmap')
         dataset.to_device(torch.device('cpu'))
         dataset.shuffle_points()

         val_sample  = int(len(dataset) * hparams['val_ratio'])
         test_sample = int(len(dataset) * (1-hparams['dataset_ratio']))
         train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [len(dataset) - val_sample -test_sample, val_sample, test_sample])
         
         train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
         val_loader   = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)


         # MODEL CREATION
         agent = DualPerceiver(**hparams)
         agent.to(device=torch.device('cuda'))
         ### LOG ##
         run_id = folder_name + '-'  + str(time.strftime("%m-%d-%H-%M"))
         wandb.init(project="cloth_attention_ablation", name=str(run_id), config=hparams)
         writer = SummaryWriter(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', run_id))
         writer.add_text('Hyperparameters', str(hparams))
         writer.flush()


         ###### 2) TRAIN POINT #####
         for epoch in tqdm(range(hparams['num_epochs']), desc='Point training'):

            epoch_train_result = agent.trainer_pointprediction(train_loader)
            if write_log:
               for key, value in epoch_train_result.items():
                  wandb.log({f'Train/{key}': value}, step=epoch)
                  writer.add_scalar(f'Train/{key}', value, epoch)
               writer.flush()

            epoch_val_result = agent.validate_pointprediction(val_loader)
            if write_log:
               for key, value in epoch_val_result.items():
                  wandb.log({f'Val/{key}': value}, step=epoch)
                  writer.add_scalar(f'Val/{key}', value, epoch)
               writer.flush()  
         if save_model :
            model_path = os.path.join(save_model_path, str(run_id)+'.pth')
            agent.save_best_model(model_path)
            
            with open(os.path.join(save_model_path, f'{run_id}_hparams.pickle'), 'wb') as f:
               pickle.dump(hparams, f, protocol=4)


         ###### 3) TRAIN ACTION #####
         for epoch in tqdm(range(hparams['num_epochs']), desc='Action training'):

            epoch_train_result = agent.trainer_action(train_loader)
            if write_log:
               for key, value in epoch_train_result.items():
                  wandb.log({f'Train/{key}': value}, step=epoch)
                  writer.add_scalar(f'Train/{key}', value, epoch)
               writer.flush()
            epoch_val_result = agent.validate_action(val_loader)
            
            if write_log:
               for key, value in epoch_val_result.items():
                  wandb.log({f'Val/{key}': value}, step=epoch)
                  writer.add_scalar(f'Val/{key}', value, epoch)
               writer.flush()

         if save_model :
            model_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model', folder_name)
            model_path = os.path.join(model_base_path, str(run_id)+'.pth')
            agent.save_best_model(model_path)
            
            with open(os.path.join(model_base_path, f'{run_id}_hparams.pickle'), 'wb') as f:
               pickle.dump(hparams, f, protocol=4)

      except RuntimeError as e:
         print('Fail to load the configuration')
         print(e)

      wandb.finish()