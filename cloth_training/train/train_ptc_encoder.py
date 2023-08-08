


import torch
from cloth_training.model.model_architecture.dataset_gen import GymClothDataset
from cloth_training.model.pointcloud_encoder import PointCloudEncoder

import os, pickle, time
from tqdm import tqdm
from torch.utils.data import DataLoader

if __name__ == '__main__' :

   folder_name = 'ptc_encoder'
   write_log = True
   save_model = True


   #load hparams from file
   hyperparameters = [
                     {
                     'num_epochs' : 300,
                     'val_ratio' : 0.2,
                     'batch_size' : 64,
                     'lr' : 0.001,
                     'depth' : 1,
                     'input_latent_dim' : 625,
                     'input_embedding_dim' : 64,
                     'num_latent_heads' : 4,
                     'num_cross_heads' : 4,
                     'num_latent_layers' : 3,
                     'seed' : 42
                     },
                     {
                     'num_epochs' : 300,
                     'val_ratio' : 0.2,
                     'batch_size' : 128,
                     'lr' : 0.001,
                     'depth' : 1,
                     'input_latent_dim' : 625,
                     'input_embedding_dim' : 64,
                     'num_latent_heads' : 4,
                     'num_cross_heads' : 4,
                     'num_latent_layers' : 3,
                     'seed' : 42
                     },
                     {
                     'num_epochs' : 300,
                     'val_ratio' : 0.2,
                     'batch_size' : 64,
                     'lr' : 0.001,
                     'depth' : 3,
                     'input_latent_dim' : 625,
                     'input_embedding_dim' : 64,
                     'num_latent_heads' : 4,
                     'num_cross_heads' : 4,
                     'num_latent_layers' : 3,
                     'seed' : 42
                     },
                     {
                     'num_epochs' : 300,
                     'val_ratio' : 0.2,
                     'batch_size' : 128,
                     'lr' : 0.001,
                     'depth' : 3,
                     'input_latent_dim' : 625,
                     'input_embedding_dim' : 64,
                     'num_latent_heads' : 4,
                     'num_cross_heads' : 4,
                     'num_latent_layers' : 3,
                     'seed' : 42
                     },
                     ]





   #for hparams in tqdm(iterate_hyperparameters(hyperparameters), desc='Hyperparameter Training'):

   for hparams in hyperparameters :
      print(f'Start run with hyperparameters: \n {hparams}')


      # Iterating over all combinations of hyperparameters
      dataset = torch.load('/home/kgalassi/code/cloth/cloth_training/cloth_training/dagger/pull/pull_dataset.pt')
      dataset.set_obs_type('ptc')
      dataset.set_output_type('ptc')
      dataset.to_device(torch.device('cpu'))
      dataset.shuffle_points()

      val_sample  = int(len(dataset) * hparams['val_ratio'])
      train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_sample, val_sample])
      
      train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
      val_loader   = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)

      try :

         # MODEL CREATION
         agent = PointCloudEncoder(**hparams)
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
         agent.reset_train()
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

      except Exception as e:
         print(e)

         continue
