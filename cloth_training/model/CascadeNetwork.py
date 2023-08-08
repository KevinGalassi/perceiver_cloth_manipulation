import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os

import pickle

from cloth_training.model.model_architecture.base_model import InputEncoder, ProbDecoder
from cloth_training.model.model_architecture.base_model import OffsetNetwork3stage, OffsetNetworkDualCat
from cloth_training.model.model_architecture.base_model import OffsetNetworkECatFc, OffsetNetworkDualEDualCat
from cloth_training.model.model_architecture.base_model import OffsetNetworkObsEncodingDualCat, OffsetNetworkObsEDualCat
from cloth_training.model.model_architecture.base_model import OffsetNetworkTriple
from cloth_training.model.model_architecture.model_utils import get_precision_at_k

from cloth_training.model.model_architecture.model_utils import create_dual_offset_plot
from cloth_training.model.model_architecture.model_utils import get_angle 

import numpy as np
import matplotlib.pyplot as plt

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



class CascadeNetwork(nn.Module):

   def __init__(self, **kwargs) -> None:
      '''
      Kwargs list :
      - no_points : number of points in the cloth (default: 400)
      - action_dim : number of action dimension _offset_ (default: 2)
      - hidden_dim : number of hidden dimension (default: 128)
      '''

      super().__init__()

      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      no_points         = int(kwargs.get('no_points', 441))
      action_dim        = int(kwargs.get('action_dim', 2))
      hidden_dim        = int(kwargs.get('hidden_dim', 128))
      offset_hidden_dim = int(kwargs.get('offset_hidden_dim', 8))
      inj_hidden_dim    = int(kwargs.get('inj_hidden_dim', 8))
      self.lr             = kwargs.get('lr', 1e-3)
      offset_network_type = kwargs.get('offset_network_type', 0)
      use_position        = kwargs.get('use_position', True)

      obs_dim    = int(no_points * 3)
      output_dim = int(no_points)

      if kwargs.get('seed', None) is not None:
         set_seed(kwargs.get('seed'))


      self.init_params = {'no_points': no_points,
                           'action_dim': action_dim,
                           'hidden_dim': hidden_dim,
                           'offset_hidden_dim': offset_hidden_dim,
                           'inj_hidden_dim': inj_hidden_dim,
                           'offset_network_type' : offset_network_type,
                           'use_position': use_position,
                           'lr': self.lr}
      
      self.encoder = InputEncoder(obs_dim, hidden_dim)
      self.decoder_prob = ProbDecoder(hidden_dim, output_dim)

      self.use_position = use_position
      if use_position :
         cat_dim = 3
      else :
         cat_dim = no_points


      if offset_network_type == 0:
         self.decoder_offset = OffsetNetwork3stage(hidden_dim, cat_dim)
      elif offset_network_type == 1:
         self.decoder_offset = OffsetNetworkECatFc(hidden_dim, cat_dim)
      elif offset_network_type == 2:
         self.decoder_offset = OffsetNetworkTriple(obs_dim, offset_hidden_dim, cat_dim, obs_dim, False)
      elif offset_network_type == 3:
         self.decoder_offset = OffsetNetworkDualEDualCat(hidden_dim, offset_hidden_dim, cat_dim, inj_hidden_dim)
      elif offset_network_type == 4:
         self.decoder_offset = OffsetNetworkObsEncodingDualCat(hidden_dim, offset_hidden_dim, cat_dim, inj_hidden_dim, obs_dim)
      elif offset_network_type == 5:
         self.decoder_offset = OffsetNetworkObsEDualCat(obs_dim, offset_hidden_dim, cat_dim, inj_hidden_dim, obs_dim, True)
      elif offset_network_type == 6:
         self.decoder_offset = OffsetNetworkDualCat(hidden_dim, offset_hidden_dim, cat_dim, inj_hidden_dim)
      else :
         raise ValueError('Invalid offset network type')




      self.mse_loss = nn.MSELoss()

      # Freeze parameters of encoder and prob decoder
      for param in self.encoder.parameters():
         param.requires_grad = False
      for param in self.decoder_prob.parameters():
         param.requires_grad = False

      # Offset network parameters
      offset_params = list(self.decoder_offset.parameters())
      self.offset_optimizer = optim.AdamW(offset_params, lr=self.lr)
      self.offset_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.offset_optimizer, 'min')



      self.best_model = {}

      self.best_prob_network_loss = float('inf')
      self.best_offset_network_loss = float('inf')
      
   
   def forward(self, x) :
   
      h = self.encoder(x)

      p = self.decoder_prob(h)

      #p_s = p.argmax(dim=1)
      argmax_indices = torch.argmax(p, dim=1)

      # Create second tensor with 0 and 1 at argmax indices
      if self.use_position :
         id_one = p.argmax()*3

         pos = torch.cat((x[torch.arange(0, x.shape[0]), id_one].unsqueeze(1),
                        x[torch.arange(0, x.shape[0]), id_one+1].unsqueeze(1),
                        x[torch.arange(0, x.shape[0]), id_one+2].unsqueeze(1)), dim=1)

         a = self.decoder_offset(h, pos)
      else :
         a = self.decoder_offset(h, F.softmax(p))
      return p, a



   def forward_encoder(self, x) :
      return self.encoder(x)
   

   def forward_prob_network(self, x) :
      return self.decoder_prob(x)
   
   def forward_offset_network(self, x, p) :
      return self.decoder_offset(x, p)


   def trainer(self, train_loader):
      for param in self.encoder.parameters():
         param.requires_grad = False
      for param in self.decoder_prob.parameters():
         param.requires_grad = False

      self.train()
      total_train_loss = 0.0
      distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0

      for i , data in enumerate(train_loader) :
      #for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Progress'):

         # get the inputs and labels from the data loader
         obs, gt = data


         # forward + backward + optimize
         h = self.encoder(obs)
         p = self.decoder_prob(h)         
         
         # Create second tensor with 0 and 1 at argmax indices
         if self.use_position :           
            id_one = (torch.where(gt[:, :-2] == 1)[1])*3

            pos = torch.cat((obs[torch.arange(0, obs.shape[0]), id_one].unsqueeze(1),
                              obs[torch.arange(0, obs.shape[0]), id_one+1].unsqueeze(1),
                              obs[torch.arange(0, obs.shape[0]), id_one+2].unsqueeze(1)), dim=1)
            a = self.decoder_offset(h, pos)

         else :
            p = F.softmax(p, dim=1)
            a = self.decoder_offset(h, p)


         # Get loss function
         loss = self.mse_loss(a, gt[:, -2:])

         # Distance
         distance += torch.sqrt(torch.sum((a - gt[:, -2:])**2, dim=1)).sum().item() / len(gt)
         distance_x += torch.sqrt(torch.sum((a[:,0] - gt[:, -2])**2, dim=0)).sum().item() / len(gt)
         distance_y += torch.sqrt(torch.sum((a[:,1] - gt[:, -1])**2, dim=0)).sum().item() / len(gt)
         angle += torch.sqrt(torch.sum((torch.atan2(a[:,1], a[:,0]) - torch.atan2(gt[:, -1], gt[:, -2]))**2, dim=0)).sum().item() / len(gt)

         # zero the parameter gradients
         self.offset_optimizer.zero_grad()
         loss.backward()
         self.offset_optimizer.step()

         total_train_loss += loss.item()

         # Log learning rate to Tensorboard
         lr = self.offset_optimizer.param_groups[0]["lr"]

      self.training_step = {'train_loss': total_train_loss / len(train_loader),
                           'distance': distance / len(train_loader),
                           'distance_x': distance_x / len(train_loader),
                           'distance_y': distance_y / len(train_loader),
                           'angle': angle / len(train_loader),
                           'lr': lr}
      return self.training_step

   def validate(self, val_loader):

      total_val_loss = 0.0
      distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0

      self.eval()
      with torch.no_grad():
         for i , data in enumerate(val_loader) :

            # get the inputs and labels from the data loader
            obs, gt = data
            
            h = self.encoder(obs)
            p = self.decoder_prob(h)

            # Create second tensor with 0 and 1 at argmax indices
            if self.use_position :
               id_one = (torch.where(gt[:, :-2] == 1)[1])*3

               pos = torch.cat((obs[torch.arange(0, obs.shape[0]), id_one].unsqueeze(1),
                                 obs[torch.arange(0, obs.shape[0]), id_one+1].unsqueeze(1),
                                 obs[torch.arange(0, obs.shape[0]), id_one+2].unsqueeze(1)), dim=1)
               a = self.decoder_offset(h, pos)

            else :
               p = F.softmax(p, dim=1)
               a = self.decoder_offset(h, p)

            # Get loss function
            loss = self.mse_loss(a, gt[:, -2:])

            #Get accuracy
            distance += torch.sqrt(torch.sum((a - gt[:, -2:])**2, dim=1)).sum().item() / len(gt)
            distance_x += torch.sqrt(torch.sum((a[:,0] - gt[:, -2])**2, dim=0)).sum().item() / len(gt)
            distance_y += torch.sqrt(torch.sum((a[:,1] - gt[:, -1])**2, dim=0)).sum().item() / len(gt)
            angle += torch.sqrt(torch.sum((torch.atan2(a[:,1], a[:,0]) - torch.atan2(gt[:, -1], gt[:, -2]))**2, dim=0)).sum().item() / len(gt)


            # Sum loss
            total_val_loss += loss.item()

         # zero the parameter gradients
         self.offset_scheduler.step(total_val_loss)


         self.offset_val_step = {'val_loss': total_val_loss / len(val_loader),
                                 'distance': distance / len(val_loader),
                                 'distance_x': distance_x / len(val_loader),
                                 'distance_y': distance_y / len(val_loader),
                                 'angle': angle / len(val_loader),
                                                     }
         
               # Save Best Model
         if total_val_loss/ len(val_loader) < self.best_offset_network_loss:
            self.best_offset_network_loss = total_val_loss/ len(val_loader)
            self.best_model['encoder'] = self.encoder.state_dict()
            self.best_model['decoder_prob'] = self.decoder_prob.state_dict()
            self.best_model['decoder_offset'] = self.decoder_offset.state_dict()
            self.best_model['prob_network_loss'] = self.best_offset_network_loss

         return self.offset_val_step
      

   def test_network(self, test_loader, verbose=False) :

      distance = 0.0
      acc_p, precision_at_5, precision_at_3 = 0.0, 0.0, 0.0

      angles = torch.empty(len(test_loader))
      dx = torch.empty(len(test_loader))
      dy = torch.empty(len(test_loader))


      self.eval()

      distance_list = []
      accuracy_prob_list = []
      
      distance_list = []
      accuracy_prob_list = []
      precision_at_5_list = []
      precision_at_3_list = []

      with torch.no_grad():
         for i , data in enumerate(test_loader) :

            # get the inputs and labels from the data loader
            obs, gt = data


            if obs.dim() == 1:  # Check if batch size is 1
               obs = obs.unsqueeze(0)
               gt = gt.unsqueeze(0)
               
            h = self.encoder(obs)
            p = self.decoder_prob(h)

   
            argmax_indices = torch.argmax(p, dim=1)
            p_s = torch.zeros_like(p)
            p_s[torch.arange(p.size(0)), argmax_indices] = 1

            #a = self.decoder_offset(h, p_s)
            a = self.decoder_offset(h, gt[:, :-2])
            #a = torch.clamp(a, 0, 1)
            # forward two models
            #p, a = self.forward(obs)

            #Distance
            distance += torch.sqrt(torch.sum((a - gt[:, -2:])**2, dim=1)).sum().item() / len(gt)
            dx[i] = gt[:, -2] - a[:, 0] 
            dy[i] = gt[:, -1] - a[:, 1]

            # get angle between predicted and ground truth
            angles[i] = get_angle(a, gt[:, -2:])



            #Get accuracy probability
            acc_p += (torch.argmax(p, dim=1) == torch.argmax(gt[:,:-2], dim=1)).sum().item() / len(gt)

            #Get precision@k
            precision_at_5 += get_precision_at_k(5, p, gt)
            precision_at_3 += get_precision_at_k(3, p, gt)

            
            distance_list.append(torch.sqrt(torch.sum((a - gt[:, -2:])**2, dim=1)).sum().item() / len(gt))
            accuracy_prob_list.append((torch.argmax(p, dim=1) == torch.argmax(gt[:,:-2], dim=1)).sum().item() / len(gt))
            precision_at_3_list.append(get_precision_at_k(3, p, gt))
            precision_at_5_list.append(get_precision_at_k(5, p, gt))

            if True:
               print(f'Preidiction {i}/ {len(test_loader)}')
               print('Predicted offset : ', a.tolist())
               print('Ground truth offset     : ', gt[:, -2:].tolist())

               folder = '06-01-18-08'
               base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), folder, f'prediction_{i}.png')   
               #create_combined_plot(obs, a, gt, p, base_path)
               create_dual_offset_plot(obs, a, gt, base_path)

               print('\n\n')

         # Count how many time error is below threshold

         angles = angles*180/3.14

         self.test_step = {'distance' : distance_list,
                           'accuracy_prob' : accuracy_prob_list,
                           'precision5': precision_at_5_list,
                           'precision3': precision_at_3_list,
                           'angles' : angles,
                           'dx' : dx,
                           'dy' : dy,
                           }
         



         return self.test_step




 

   def load_model(self, encoder, decoder_prob, decoder_offset):
      self.encoder.load_state_dict(encoder)
      self.decoder_prob.load_state_dict(decoder_prob)
      self.decoder_offset.load_state_dict(decoder_offset)

   def save_model(self, path): 
      state = {'init_params': self.init_params,
                    'encoder': self.encoder.state_dict(),
                     'decoder_prob': self.decoder_prob.state_dict(),
                     'decoder_offset': self.decoder_offset.state_dict(),
      }

      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)
         
      torch.save(state, path)


   def save_best_model(self, path):

      state = {'init_params': self.init_params,
                    'encoder': self.best_model['encoder'],
                     'decoder_prob': self.best_model['decoder_prob'],
                     'decoder_offset': self.best_model['decoder_offset'],
                     'prob_network_loss': self.best_prob_network_loss,
                     'offset_network_loss': self.best_offset_network_loss,
      }

      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)
         
      torch.save(state, path)

   
