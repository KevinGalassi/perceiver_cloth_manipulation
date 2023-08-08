



import torch
from cloth_training.model.model_architecture.dataset_gen import GymClothDataset
from cloth_training.model.perceiver_heatmap import HeatPerceiver

import os, pickle, sys, time
from tqdm import tqdm
from torch.utils.data import DataLoader


from cloth_training.model.model_architecture.model_utils import iterate_hyperparameters, set_seed
from cloth_training.train.utils import trainer

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == '__main__' :


   folder_name = 'heat5'
   write_log = True
   save_model = True


   #load hparams from file


   hyperparameters = [
                        {'num_epochs' : 2,
                        'batch_size' : 8,            
                        'val_ratio'  : 0.03,
                        'dataset_ratio' : 0.95,
                        'seed' : 42,

                        #Heat Predictor Perceiver
                        'depth' : 1,
                        'input_dim' : 3,
                        'input_embedding_dim' : 128,
                        'num_latents' : 50,
                        'num_cross_heads' : 4,
                        'num_output_heads' : 4,
                        'num_latent_heads' : 4,
                        'num_latent_layers' : 3,
                        'lr_heat' : 1e-4,

                        #Point Network
                        'point_latent_heads' : 4,
                        'point_latent_layers' : 3,
                        'lr_point' : 1e-4,

                        # Action parameters
                        'action_latent_heads' : 4,
                        'action_cross_heads' : 4,
                        'action_latent_layers' : 3,
                        'lr_action' : 1e-4,
                        },
                     ]




   #for hparams in tqdm(iterate_hyperparameters(hyperparameters), desc='Hyperparameter Training'):

   for hparams in hyperparameters :
      print(f'Start run with hyperparameters: \n {hparams}')


      # Iterating over all combinations of hyperparameters
      dataset = torch.load('/home/kgalassi/code/cloth/cloth_training/cloth_training/dataset/pull/pull_dataset.pt')
      dataset.set_obs_type('heatmap')
      dataset.set_output_type('heatmap')
      dataset.to_device(torch.device('cpu'))
      dataset.shuffle_points()

      val_sample  = int(len(dataset) * hparams['val_ratio'])
      test_sample = int(len(dataset) * (1-hparams['dataset_ratio']))
      train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [len(dataset) - val_sample -test_sample, val_sample, test_sample])
      
      train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
      val_loader   = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)


      try :
         # MODEL CREATION
         agent = HeatPerceiver(**hparams)
         agent.to(device=torch.device('cuda'))
         ### LOG ##
         log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', folder_name)
         model_base_path   = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model', folder_name)
         run_id = time.strftime("%m-%d-%H-%M")


         if write_log:
            from torch.utils.tensorboard import SummaryWriter
            print('Logging to ', os.path.join(log_path, str(run_id)))
            writer = SummaryWriter(log_dir=os.path.join(log_path, str(run_id)))
            writer.add_text('Hyperparameters', str(hparams))
            writer.flush()
            

         ###### 1) TRAIN HEATMAP #####
         for epoch in tqdm(range(hparams['num_epochs']), desc='Epoch training'):

            epoch_train_result = agent.trainer_heat(train_loader)
            if write_log :
               for key, value in epoch_train_result.items():
                  writer.add_scalar(f'TrainHeat/{key}', value, epoch)
               writer.flush()

            epoch_val_result = agent.validate_heat(val_loader)

            if write_log :
               for key, value in epoch_val_result.items():
                  writer.add_scalar(f'ValHeat/{key}', value, epoch)
               writer.flush()

            if save_model :
               model_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model', folder_name)
               model_path = os.path.join(model_base_path, str(run_id)+'.pth')
               agent.save_best_model(model_path)
               
               with open(os.path.join(model_base_path, f'{run_id}_hparams.pickle'), 'wb') as f:
                  pickle.dump(hparams, f, protocol=4)


         ###### 2) TRAIN POINT #####
         for epoch in tqdm(range(hparams['num_epochs']), desc='Point training'):

            epoch_train_result = agent.trainer_pointprediction(train_loader)
            if write_log :
               for key, value in epoch_train_result.items():
                  writer.add_scalar(f'TrainPoint/{key}', value, epoch)
               writer.flush()

            epoch_val_result = agent.validate_pointprediction(val_loader)

            if write_log :
               for key, value in epoch_val_result.items():
                  writer.add_scalar(f'ValPoint/{key}', value, epoch)
               writer.flush()

            if save_model :
               model_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model', folder_name)
               model_path = os.path.join(model_base_path, str(run_id)+'.pth')
               agent.save_best_model(model_path)
               
               with open(os.path.join(model_base_path, f'{run_id}_hparams.pickle'), 'wb') as f:
                  pickle.dump(hparams, f, protocol=4)

         ###### 3) TRAIN ACTION #####
         for epoch in tqdm(range(hparams['num_epochs']), desc='Action training'):

            epoch_train_result = agent.trainer_action(train_loader)
            if write_log :
               for key, value in epoch_train_result.items():
                  writer.add_scalar(f'TrainAction/{key}', value, epoch)
               writer.flush()

            epoch_val_result = agent.validate_action(val_loader)

            if write_log :
               for key, value in epoch_val_result.items():
                  writer.add_scalar(f'ValAction/{key}', value, epoch)
               writer.flush()

            if save_model :
               model_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model', folder_name)
               model_path = os.path.join(model_base_path, str(run_id)+'.pth')
               agent.save_best_model(model_path)
               
               with open(os.path.join(model_base_path, f'{run_id}_hparams.pickle'), 'wb') as f:
                  pickle.dump(hparams, f, protocol=4)

      except RuntimeError as e:
         print('Fail to load the configuration, continu....')