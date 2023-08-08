import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
def set_seed(seed):
   if seed == None :
      print('Warning : seed is None')
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

from tqdm import tqdm
import os

from cloth_training.model.model_architecture.base_model import InputEncoder
from cloth_training.model.model_architecture.base_model import OffsetNetworkObsEDualCat

import matplotlib.pyplot as plt


class OffsetEncoderNetwork(nn.Module):

   def __init__(self,device=None, **kwargs) -> None:
      '''
      Kwargs list :
      - input_dim : number of points in the cloth (default: 400)
      - action_dim : number of action dimension _offset_ (default: 2)
      - hidden_dim : number of hidden dimension (default: 128)
      '''

      super().__init__()

      if device is None:
         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      else :
         self.device = device

      input_dim         = int(kwargs.get('input_dim', 400))
      inj_hidden_dim    = int(kwargs.get('inj_hidden_dim', 8))
      middle_dim        = int(kwargs.get('middle_dim', 128))
      cat_dim           = int(kwargs.get('cat_dim', 3))
      self.lr           = kwargs.get('lr', 1e-3)

      if kwargs.get('seed', None) is not None:
         set_seed(kwargs.get('seed'))

      self.init_params = {'input_dim': input_dim,
                           'inj_hidden_dim': inj_hidden_dim,
                           'middle_dim': middle_dim,
                           'cat_dim': cat_dim,
                           'lr': self.lr,
                           'seed': kwargs.get('seed', None)}
      
      self.decoder_offset = OffsetNetworkObsEDualCat(input_dim,
                                                      middle_dim, 
                                                      cat_dim, 
                                                      inj_hidden_dim)




      self.mse_loss = nn.MSELoss()



      self.params = list(self.decoder_offset.parameters())
      self.optimizer = optim.AdamW(self.params, lr=self.lr, weight_decay=1e-5)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

      self.best_model = {}
      self.best_val_loss = float('inf')
      
   
   def forward(self, obs, p) :
   
      
      a = self.decoder_offset(obs, p)
  
      return a.relu()


   def trainer(self, train_loader):

      self.best_prob_network_loss = 100000
      self.params = list(self.decoder_offset.parameters())
      self.optimizer = optim.AdamW(self.params, lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')


      self.train()
      total_train_loss = 0.0
      distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0

      for i , data in enumerate(train_loader) :

         # get the inputs and labels from the data loader
         obs, gt = data

         b = obs.shape[0]
         obs_flatten = obs.reshape(b, -1)
         gt_prob = gt[0]
         gt_action = gt[1]
         # forward + backward + optimize
         
         id_one = gt_prob.argmax(dim=1)
         pos = obs[torch.arange(0, b), id_one]


         obs_flatten = obs_flatten.to(self.device)
         gt_prob = gt_prob.to(self.device)
         gt_action = gt_action.to(self.device).to(torch.float32)
         pos = pos.to(self.device)

         a = self.decoder_offset(obs_flatten, pos)


         print(a.shape)
         print(gt_action.shape)

         print(a.dtype)
         print(gt_action.dtype)

         # Get loss function
         loss = self.mse_loss(a, gt_action)
         #a = torch.clamp(a, 0, 1)

         #Get accuracy
         distance += torch.sqrt(torch.sum((a - gt_action)**2, dim=1)).sum().item() / b
         distance_x += torch.sqrt(torch.sum((a[:,0] - gt_action[:,0])**2, dim=0)).sum().item() / b
         distance_y += torch.sqrt(torch.sum((a[:,1] - gt_action[:,1])**2, dim=0)).sum().item() / b
         angle += torch.sqrt(torch.sum((torch.atan2(a[:,1], a[:,0]) - torch.atan2(gt_action[:,1], gt_action[:,0]))**2, dim=0)).sum().item() / b

         # zero the parameter gradients
         self.optimizer.zero_grad()
         loss.backward()
         self.optimizer.step()

         total_train_loss += loss.item()

         # Log learning rate to Tensorboard
         lr = self.optimizer.param_groups[0]["lr"]

      self.training_step = {'train_loss': total_train_loss / len(train_loader),
                           'distance': distance / len(train_loader),
                           'distance_x': distance_x / len(train_loader),
                           'distance_y': distance_y / len(train_loader),
                           'angle': angle / len(train_loader),                           'lr': lr}
      return self.training_step

   def validate(self, val_loader):

      total_val_loss = 0.0
      distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0

      self.eval()
      with torch.no_grad():
         for i , data in enumerate(val_loader) :

            obs, gt = data

            b = obs.shape[0]
            obs_flatten = obs.reshape(b, -1)
            gt_prob = gt[0]
            gt_action = gt[1]
            
            id_one = gt_prob.argmax(dim=1)
            pos = obs[torch.arange(0, b), id_one]
            

            obs_flatten = obs_flatten.to(self.device)
            gt_prob = gt_prob.to(self.device)
            gt_action = gt_action.to(self.device).to(torch.float32)
            pos = pos.to(self.device)


            a = self.decoder_offset(obs_flatten, pos)

            # Get loss function
            loss = self.mse_loss(a, gt_action)

            #Get accuracy
            distance += torch.sqrt(torch.sum((a - gt_action)**2, dim=1)).sum().item() / b
            distance_x += torch.sqrt(torch.sum((a[:,0] - gt_action[:,0])**2, dim=0)).sum().item() / b
            distance_y += torch.sqrt(torch.sum((a[:,1] - gt_action[:,1])**2, dim=0)).sum().item() / b
            angle += torch.sqrt(torch.sum((torch.atan2(a[:,1], a[:,0]) - torch.atan2(gt_action[:,1], gt_action[:,0]))**2, dim=0)).sum().item() / b

            # Sum loss
            total_val_loss += loss.item()



         self.scheduler.step(total_val_loss)
         self.offset_val_step = {'val_loss': total_val_loss / len(val_loader),
                                 'distance': distance / len(val_loader),
                                 'distance_x': distance_x / len(val_loader),
                                 'distance_y': distance_y / len(val_loader),
                                 'angle': angle / len(val_loader),
                                }
         
               # Save Best Model
         if total_val_loss/ len(val_loader) < self.best_val_loss:
            self.best_val_loss = total_val_loss/ len(val_loader)
            self.best_model['decoder_offset'] = self.decoder_offset.state_dict()
            self.best_model['off_network_loss'] = self.best_val_loss

         return self.offset_val_step
      


   def save_model(self, path):
      data = {'init_params': self.init_params,
               'model': self.decoder_offset.state_dict(),
      }
      
      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)


      torch.save(data, path)




   def save_best_model(self, path):
      data = {'init_params': self.init_params,
               'model': self.best_model['decoder_offset'],
               'offset_network_loss': self.best_val_loss,
      }
      
      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)


      torch.save(data, path)

   
