import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

from tqdm import tqdm

from cloth_training.model.model_architecture.model_utils import get_precision_at_k
from cloth_training.model.model_architecture.attention_models import PerceiverIOv2, OffsetPerceiver
from cloth_training.model.model_architecture.model_utils import Lamb

import numpy as np
from cloth_training.model.model_architecture.model_utils import set_seed

import pickle, os



class PerceiverOffset(nn.Module):

   def __init__(self, **kwargs) -> None:
      '''
      Kwargs list :
      - no_points : number of points in the cloth (default: 400)
      - action_dim : number of action dimension _offset_ (default: 2)
      - hidden_dim : number of hidden dimension (default: 128)
      '''

      #super(TransformerModel, self).__init__()

      super().__init__()

      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      depth = int(kwargs.get('depth', 3))
      input_dim  = int(kwargs.get('input_dim', 3))
      input_embedding_dim = int(kwargs.get('input_embedding_dim', 128))
      output_dim  = int(kwargs.get('output_dim', 441))
      num_latents = int(kwargs.get('num_latents', 50))
      num_cross_heads   = int(kwargs.get('num_cross_heads', 8))
      num_output_heads  = int(kwargs.get('num_output_heads', 8))
      num_latent_heads  = int(kwargs.get('num_latent_heads', 8))
      num_latent_layers  = int(kwargs.get('num_latent_layers', 3))
      self.lr          = kwargs.get('lr', 1e-4)
      seed        = kwargs.get('seed', None)

      if seed is not None:
         set_seed(seed)

      self.init_params = {'depth' : depth,
                           'input_dim' : input_dim,
                           'input_embedding_dim' : input_embedding_dim,
                           'output_dim' : output_dim,
                           'num_latents' : num_latents,
                           'num_cross_heads' : num_cross_heads,
                           'num_output_heads' : num_output_heads,
                           'num_latent_heads' : num_latent_heads,
                           'num_latent_layers' : num_latent_layers,
                           'lr' : self.lr,
                           'seed' : seed,
                           'networktype' : 'PerceiverIO',
                        }

      self.prob_model = PerceiverIOv2(depth                 = depth,
                                 input_dim           = input_dim,          # C
                                 input_embedding_dim = input_embedding_dim,# D
                                 output_dim          = output_dim,         # O
                                 num_latents         = num_latents,        # M
                                 num_cross_heads     = num_cross_heads, 
                                 num_output_heads    = num_output_heads,
                                 num_latent_heads    = num_latent_heads,
                                 num_latent_layers   = num_latent_layers, 
                                 seed = seed,
                                 )
      

      self.offset_model = OffsetPerceiver(depth            = depth,
                                          input_query_dim  = 3,          # C
                                          input_embedding_dim = input_embedding_dim,# D
                                          output_dim          = 2,         # O
                                          num_cross_heads     = num_cross_heads, 
                                          num_output_heads    = num_output_heads,
                                          num_latent_heads    = num_latent_heads,
                                          num_latent_layers   = num_latent_layers, 
                                          seed = seed,
                                          )

      self.cross_loss  = nn.CrossEntropyLoss()
      self.mse_loss = nn.MSELoss()

      self.best_model = {}



   def forward_probability(self, x):      
      p = self.prob_model(x)
      return p
   

   def forward_offset(self, p, x):
      o = self.offset_model(p, x)


   def forward(self, obs):

      p = self.forward_probability(obs)

      id_one = torch.argmax(p).item()

      pos = torch.cat((obs[torch.arange(0, obs.shape[0]), id_one].unsqueeze(1),
                        obs[torch.arange(0, obs.shape[0]), id_one+1].unsqueeze(1),
                        obs[torch.arange(0, obs.shape[0]), id_one+2].unsqueeze(1)), dim=1)

      l = self.prob_model.perceiver_layer.cross_input

      o = self.forward_offset(pos, l)

      return p, o


   def prob_network_train_init(self) :
      for param in self.prob_model.parameters():
         param.requires_grad = True

      self.params = list(self.prob_model.parameters())

      self.optimizer = Lamb(self.params, lr=self.lr)

      #self.optimizer = optim.AdamW(self.params, lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

      self.best_model = {}
      self.best_val_loss = float('inf')

   def offset_network_train_init(self) :
      for param in self.prob_model.parameters():
         param.requires_grad = False
      
      self.params = list(self.offset_model.parameters())
      self.optimizer = optim.AdamW(self.params, lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

      self.best_model = {}
      self.best_val_loss = float('inf')


   def train_probability(self, train_loader):
      self.train()
      total_train_loss = 0.0
      precision_at_4, precision_at_3,precision_at_2 = 0.0, 0.0, 0.0
      acc = 0.0

      for i, data in enumerate(train_loader) :
      #for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Progress'):
         # get the inputs and labels from the data loader
         obs, gt = data

         obs = obs.reshape(obs.shape[0], -1, 3)

         p = self.forward_probability(obs)

         p = p.reshape(gt.shape[0], -1)
         # Get loss function
         prob_loss = self.cross_loss(p, gt[:,:-2])

         # Sum the loss
         loss = prob_loss

         acc += (torch.argmax(p, dim=1) == torch.argmax(gt[:,:-2], dim=1)).sum().item() / len(gt)
         precision_at_4 += get_precision_at_k(5, p, gt)
         precision_at_3 += get_precision_at_k(3, p, gt)
         precision_at_2 += get_precision_at_k(2, p, gt)


         # zero the parameter gradients
         self.optimizer.zero_grad()
         loss.backward()
         self.optimizer.step()

         total_train_loss += loss.item()

         # Log learning rate to Tensorboard
         lr = self.optimizer.param_groups[0]["lr"]
                         
      self.training_step = {'train_loss': total_train_loss / len(train_loader),
                           'accuracy' : acc / len(train_loader),
                           'precision4': precision_at_4 / len(train_loader),
                           'precision3': precision_at_3 / len(train_loader),
                           'precision2': precision_at_2 / len(train_loader),
                           'lr': lr
                           }

      return self.training_step

   def validate_probability(self, val_loader):

      total_val_loss = 0.0
      acc, precision_at_4, precision_at_3, precision_at_2 = 0.0, 0.0, 0.0, 0.0
      self.best_prob_network_loss = float('inf')


      self.eval()
      with torch.no_grad():
         for i, data in enumerate(val_loader) :
            # get the inputs and labels from the data loader
            obs, gt = data

            # forward + backward + optimize
            obs = obs.reshape(obs.shape[0], -1, 3)
            p = self.forward_probability(obs)

            p = p.squeeze(-1)

            # Get loss function
            val_loss = self.cross_loss(p, gt[:,:-2])

            total_val_loss += val_loss.item()

            #Get accuracy
            acc += (torch.argmax(p, dim=1) == torch.argmax(gt[:,:-2], dim=1)).sum().item() / len(gt)
            precision_at_4 += get_precision_at_k(5, p, gt)
            precision_at_3 += get_precision_at_k(3, p, gt)
            precision_at_2 += get_precision_at_k(2, p, gt)

         self.scheduler.step(total_val_loss)

      self.val_step = {'val_loss': total_val_loss/ len(val_loader),
                  'accuracy' : acc/len(val_loader),
                  'precision4': precision_at_4 / len(val_loader),
                  'precision3': precision_at_3 / len(val_loader),
                  'precision2': precision_at_2 / len(val_loader),
                  }
      

      # Save Best Model
      if total_val_loss/ len(val_loader) < self.best_prob_network_loss:
         self.best_prob_network_loss = total_val_loss/ len(val_loader)
         self.best_model['prob_network'] = self.prob_model.state_dict()
         self.best_model['init_params'] = self.offset_model.state_dict()
      return self.val_step


   def train_offset(self, train_loader):
      self.train()
      total_train_loss = 0.0
      distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0

      for i , data in enumerate(train_loader) :

         # get the inputs and labels from the data loader
         obs, gt = data
         
         # Get ID of the grasped point
         prob = (torch.where(gt[:, :-2] == 1)[1])*3
         pos = torch.cat((obs[torch.arange(0, obs.shape[0]), prob].unsqueeze(1),
                          obs[torch.arange(0, obs.shape[0]), prob+1].unsqueeze(1),
                          obs[torch.arange(0, obs.shape[0]), prob+2].unsqueeze(1)), dim=1)

         # Get input cross attention query
         obs = obs.reshape(obs.shape[0], -1, 3)
         q = self.prob_model.input_cross_attention(obs)

         # get offset
         a = self.forward_offset(pos, q)


         # Get loss function
         loss = self.mse_loss(a, gt[:, -2:])
         #a = torch.clamp(a, 0, 1)

         #Get accuracy
         distance += torch.sqrt(torch.sum((a - gt[:, -2:])**2, dim=1)).sum().item() / len(gt)
         distance_x += torch.sqrt(torch.sum((a[:,0] - gt[:, -2])**2, dim=0)).sum().item() / len(gt)
         distance_y += torch.sqrt(torch.sum((a[:,1] - gt[:, -1])**2, dim=0)).sum().item() / len(gt)
         angle += torch.sqrt(torch.sum((torch.atan2(a[:,1], a[:,0]) - torch.atan2(gt[:, -1], gt[:, -2]))**2, dim=0)).sum().item() / len(gt)

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



   def validate_offset(self, val_loader):
         
      self.train()
      total_val_loss = 0.0
      distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0

      with torch.no_grad():
         for i , data in enumerate(val_loader) :

            # get the inputs and labels from the data loader
            obs, gt = data
            
            # Get ID of the grasped point
            prob = (torch.where(gt[:, :-2] == 1)[1])*3
            pos = torch.cat((obs[torch.arange(0, obs.shape[0]), prob].unsqueeze(1),
                           obs[torch.arange(0, obs.shape[0]), prob+1].unsqueeze(1),
                           obs[torch.arange(0, obs.shape[0]), prob+2].unsqueeze(1)), dim=1)

            # Get input cross attention query
            obs = obs.reshape(obs.shape[0], -1, 3)
            q = self.prob_model.input_cross_attention(obs)

            # get offset
            a = self.forward_offset(pos, q)

            # Get loss function
            loss = self.mse_loss(a, gt[:, -2:])

            #Get accuracy
            distance += torch.sqrt(torch.sum((a - gt[:, -2:])**2, dim=1)).sum().item() / len(gt)
            distance_x += torch.sqrt(torch.sum((a[:,0] - gt[:, -2])**2, dim=0)).sum().item() / len(gt)
            distance_y += torch.sqrt(torch.sum((a[:,1] - gt[:, -1])**2, dim=0)).sum().item() / len(gt)
            angle += torch.sqrt(torch.sum((torch.atan2(a[:,1], a[:,0]) - torch.atan2(gt[:, -1], gt[:, -2]))**2, dim=0)).sum().item() / len(gt)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_val_loss += loss.item()

         self.scheduler.step(total_val_loss)

         self.validation_step = {'train_loss': total_val_loss / len(val_loader),
                              'distance': distance / len(val_loader),
                              'distance_x': distance_x / len(val_loader),
                              'distance_y': distance_y / len(val_loader),
                              'angle': angle / len(val_loader),    
                                    }
         
         if total_val_loss / len(val_loader) < self.best_val_loss:
            self.best_val_loss = total_val_loss / len(val_loader)
            self.best_offset_model = self.offset_model.state_dict()

         return self.validation_step



   def save_best_model(self, path, prob=False, offset=False):
      state = {'init_params': self.init_params}

      if prob:
         state['prob_model'] = self.best_prob_model
      if offset:
         state['offset_model'] = self.best_offset_model
      
      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)

      torch.save(state, path)



   def load_model(self, prob_model = None, offset_model = None):
      
      if prob_model is not None:
         self.prob_model.load_state_dict(prob_model)
      
      if offset_model is not None:
         self.offset_model.load_state_dict(offset_model)

      return True

   def get_best_result(self):
      return {'best_train_loss': self.best_train_loss, 'best_val_loss': self.best_val_loss}
   


