import torch
from cloth_training.dataset.dataset_gen import GymClothDataset
from cloth_training.model.paper.transformer import DaggerTransformer

import os, pickle, time
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from cloth_training.model.common.model_utils import set_seed
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__' :

   folder_name = 'base_transformer'
   write_log = True
   save_model = True
   save_model_path = './saved_model'
   dataset_path = './dataset/ablation/ablation.pt'

   #load hparams from file


   hyperparameters = [
                        {'num_epochs' : 300,
                        'batch_size' : 128,            
                        'val_ratio'  : 0.2,
                        'seed' : 42,

                        #Model
                        'depth' : 1,
                        
                        'input_embedding_dim' : 128,
                        'num_latent_heads' : 4,
                        'num_output_heads' : 4,
                        'lr' : 1e-4,
                        },
                        {'num_epochs' : 300,
                        'batch_size' : 128,            
                        'val_ratio'  : 0.2,
                        'seed' : 42,

                        #Model
                        'depth' : 3,
                        'input_embedding_dim' : 128,
                        
                        'num_latent_heads' : 4,
                        'num_output_heads' : 4,
                        'lr' : 1e-4,
                        },
                        {
                           'num_epochs' : 300,
                        'batch_size' : 128,            
                        'val_ratio'  : 0.2,
                        'seed' : 42,

                        #Model
                        'depth' : 3,
                        
                        'input_embedding_dim' : 128,
                        'num_latent_heads' : 8,
                        'num_output_heads' : 8,
                        'lr' : 1e-4,
                        },
                     ]


   #for hparams in tqdm(iterate_hyperparameters(hyperparameters), desc='Hyperparameter Training'):

   for hparams in hyperparameters :
      try :
         print(f'Start run with hyperparameters: \n {hparams}')

         set_seed(hparams['seed'])
         # Iterating over all combinations of hyperparameters
         dataset = torch.load(dataset_path)
         dataset.set_obs_type('heatmap')
         dataset.set_output_type('heatmap')
         dataset.to_device(torch.device('cpu'))
         #dataset.shuffle_points()

         val_sample  = int(len(dataset) * hparams['val_ratio'])
         train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_sample, val_sample])
         
         train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
         val_loader   = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)


         # MODEL CREATION
         agent = DaggerTransformer(**hparams)
         agent.to('cuda:0')
         ### LOG ##
         run_id = folder_name + '-'  + str(time.strftime("%m-%d-%H-%M"))
         wandb.init(project="cloth_attention_ablation", name=str(run_id), config=hparams)
         writer = SummaryWriter(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', folder_name, run_id))
         writer.add_text('Hyperparameters', str(hparams))
         writer.flush()

         ###### 1) TRAIN TRANSFORMER #####
         agent.reset_train()
         for epoch in tqdm(range(hparams['num_epochs']), desc='Epoch training'):
            epoch_train_result = agent.trainer(train_loader)
            if write_log:
               for key, value in epoch_train_result.items():
                  wandb.log({f'Train/{key}': value}, step=epoch)
                  writer.add_scalar(f'Train/{key}', value, epoch)
               writer.flush()
            epoch_val_result = agent.validate(val_loader)
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
      except Exception as e:
         print(e)
         continue

      wandb.finish()