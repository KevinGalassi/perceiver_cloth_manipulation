import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from cloth_training.model.model_architecture.model_utils import get_precision_at_k
from cloth_training.model.model_architecture.attention_models import Perceiver, PerceiverIO
import numpy as np
from cloth_training.model.model_architecture.model_utils import set_seed


import pickle, os

class DaggerPerceiverIO(nn.Module):

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
      lr          = kwargs.get('lr', 1e-4)
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
                           'lr' : lr,
                           'seed' : seed,
                           'networktype' : 'PerceiverIO',
                        }

      self.model = PerceiverIO(depth                 = depth,
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
      
      self.criterion  = nn.CrossEntropyLoss()
      #self.criterion = nn.MSELoss()

      self.params = list(self.model.parameters())
      self.optimizer = optim.AdamW(self.params, lr=lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

      self.best_model = None
      self.best_val_loss = float('inf')


   def forward(self, x):
      p = self.model(x)
      return p


   

   def trainer(self, train_loader) :

      self.train()
      total_train_loss, total_prob_loss = 0.0, 0.0
      acc, precision_at_4, precision_at_3, precision_at_2, distance_prob = 0.0, 0.0, 0.0, 0.0, 0.0


      for i, data in enumerate(train_loader) :
      #for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Progress'):
         # get the inputs and labels from the data loader
         obs, gt = data

         b = obs.shape[0]
         gt_prob = gt[0]

         # forward + backward + optimize
         p = self.forward(obs)#.sigmoid()

         p = p.reshape(b, -1)
         # Get loss function
         prob_loss = self.criterion(p, gt_prob)

         # Sum the loss
         loss = prob_loss

         acc += (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
         precision_at_4 += get_precision_at_k(5, p, gt_prob)
         precision_at_3 += get_precision_at_k(3, p, gt_prob)
         precision_at_2 += get_precision_at_k(2, p, gt_prob)

         pt_taken, pt_gt = torch.argmax(p, dim=1), torch.argmax(gt_prob, dim=1)
         pt_pos, gt_pos =  obs[torch.arange(obs.shape[0]), pt_taken], obs[torch.arange(obs.shape[0]), pt_gt]
         distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b


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
                           'distance_prob': distance_prob / len(train_loader),
                           'lr': lr
                           }

      return self.training_step
   

   def validate(self, val_loader) :
      # validate the model
      total_val_loss, total_prob_loss, total_offset_loss = 0.0, 0.0, 0.0
      acc, precision_at_4, precision_at_3,precision_at_2, distance_prob = 0.0, 0.0, 0.0, 0.0, 0.0

      self.eval()
      with torch.no_grad():

         val_step = {}
         for i , data in enumerate(val_loader) :

            obs, gt = data
            b = obs.shape[0]
            obs_flatten = obs.reshape(b, -1)
            gt_prob = gt[0]          
         

            p = self.forward(obs)
            p = p.reshape(b,-1)

            # Get loss function
            prob_loss = self.criterion(p, gt_prob)
            
            total_val_loss += prob_loss.item()

            #Get accuracy
            acc += (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
            precision_at_4 += get_precision_at_k(5, p, gt)
            precision_at_3 += get_precision_at_k(3, p, gt)
            precision_at_2 += get_precision_at_k(2, p, gt)

            pt_taken, pt_gt = torch.argmax(p, dim=1), torch.argmax(gt_prob, dim=1)
            pt_pos, gt_pos =  obs[torch.arange(obs.shape[0]), pt_taken], obs[torch.arange(obs.shape[0]), pt_gt]
            distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b


      self.scheduler.step(total_val_loss)
      self.val_step = {'val_loss': total_val_loss/ len(val_loader),
                  'accuracy' : acc/len(val_loader),
                  'precision4': precision_at_4 / len(val_loader),
                  'precision3': precision_at_3 / len(val_loader),
                  'precision2': precision_at_2 / len(val_loader),
                  'distance_prob': distance_prob / len(val_loader),
                  }
      
      # Save Best Model
      if total_val_loss/ len(val_loader) < self.best_val_loss:
         self.best_train_loss = self.training_step['train_loss']
         self.best_val_loss = total_val_loss/ len(val_loader)
         self.best_model = self.model.state_dict()

         
      return val_step


   def save_best_model(self, path):

      state = {'init_params': self.init_params,
               'model': self.best_model,
      }

      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)

      torch.save(state, path)

   def load_model(self, model) :
      self.model.load_state_dict(model)
      return True

   def get_best_result(self):
      return {'best_train_loss': self.best_train_loss, 'best_val_loss': self.best_val_loss}