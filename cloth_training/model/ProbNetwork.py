import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cloth_training.model.model_architecture.model_utils import layer_init, get_precision_at_k
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

import os

import pickle

from cloth_training.model.model_architecture.base_model import InputEncoder, ProbDecoder

import matplotlib.pyplot as plt

class ProbNetwork(nn.Module):

   def __init__(self, **kwargs) -> None:
      '''
      Kwargs list :
      - no_points : number of points in the cloth (default: 400)
      - action_dim : number of action dimension _offset_ (default: 2)
      - hidden_dim : number of hidden dimension (default: 128)
      '''

      super().__init__()

      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      no_points         = int(kwargs.get('no_points'))
      action_dim        = int(kwargs.get('action_dim', 2))
      hidden_dim        = int(kwargs.get('hidden_dim', 128))
      inj_hidden_dim    = int(kwargs.get('inj_hidden_dim', 8))
      self.lr           = kwargs.get('lr', 1e-3)

      if kwargs.get('seed', None) is not None:
         set_seed(kwargs.get('seed'))

      obs_dim    = int(no_points * 3)
      output_dim = int(no_points)


      self.init_params = {'no_points': no_points,
                           'action_dim': action_dim,
                           'hidden_dim': hidden_dim,
                           'inj_hidden_dim': inj_hidden_dim,
                           'lr': self.lr,
                           'networktype' : 'EncoderProbDecoder',
                        }
      

      self.encoder = InputEncoder(obs_dim, hidden_dim)
      self.decoder_prob = ProbDecoder(hidden_dim, output_dim)


      self.prob_loss = nn.CrossEntropyLoss()
      #self.prob_loss = nn.BCELoss()
      self.mse_loss = nn.MSELoss()

      # optimizer


      # Grasp Probability parameters
      prob_params = list(self.encoder.parameters()) +\
                     list(self.decoder_prob.parameters())
      self.prob_optimizer = optim.AdamW(prob_params, lr=self.lr, weight_decay=1e-5)
      self.prob_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.prob_optimizer, 'min')

      self.best_model = {}
      self.best_prob_network_loss = float('inf')
      
   
   def forward(self, x) :
      x = x.to(self.device)
      h = self.encoder(x)
      p = self.decoder_prob(h)
      return p


   def trainer(self, train_loader):
      prob_params = list(self.encoder.parameters()) +\
                     list(self.decoder_prob.parameters())
      self.prob_optimizer = optim.AdamW(prob_params, lr=self.lr, weight_decay=1e-5)
      self.prob_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.prob_optimizer, 'min')


      self.train()
      total_train_loss = 0.0

      acc, precision_at_4, precision_at_3,precision_at_2, distance_prob = 0.0, 0.0, 0.0, 0.0, 0.0

      for i , data in enumerate(train_loader) :
         # get the inputs and labels from the data loader
         obs, gt = data

         b = obs.shape[0]
         obs_flatten = obs.reshape(b, -1)
         gt_prob = gt[0]


         # forward encoder and probability network

         obs_flatten = obs_flatten.to(self.device)
         gt_prob = gt_prob.to(self.device)
         
         h = self.encoder(obs_flatten)
         p = self.decoder_prob(h)

         # Get loss function
         if not isinstance(self.prob_loss, nn.CrossEntropyLoss):
            p = F.softmax(p, dim=1)


         loss = self.prob_loss(p, gt_prob)



         #Get accuracy
         acc += (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
         precision_at_4 += get_precision_at_k(5, p, gt_prob)
         precision_at_3 += get_precision_at_k(3, p, gt_prob)
         precision_at_2 += get_precision_at_k(2, p, gt_prob)

         pt_taken, pt_gt = torch.argmax(p, dim=1), torch.argmax(gt_prob, dim=1)
         pt_pos, gt_pos =  obs[torch.arange(b), pt_taken], obs[torch.arange(b), pt_gt]
         distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b


         # zero the parameter gradients
         self.prob_optimizer.zero_grad()
         loss.backward()
         self.prob_optimizer.step()

         total_train_loss += loss.item()

         # Log learning rate to Tensorboard
      lr = self.prob_optimizer.param_groups[0]["lr"]

      self.training_step = {'train_loss': total_train_loss / len(train_loader),
                           'accuracy' : acc / len(train_loader),
                           'precision4': precision_at_4 / len(train_loader),
                           'precision3': precision_at_3 / len(train_loader),
                           'precision2': precision_at_2 / len(train_loader),
                           'distance_prob': distance_prob / len(train_loader),
                           'lr': lr}
      return self.training_step


   def validate(self, val_loader):

      total_val_loss = 0.0
      acc, precision_at_4, precision_at_3, precision_at_2, distance_prob = 0.0, 0.0, 0.0, 0.0, 0.0

      self.eval()
      with torch.no_grad():
         for i, data in enumerate(val_loader) :
            # get the inputs and labels from the data loader
            obs, gt = data

            b = obs.shape[0]
            obs_flatten = obs.reshape(b, -1)
            gt_prob = gt[0]


            obs_flatten = obs_flatten.to(self.device)
            gt_prob = gt_prob.to(self.device)

            # forward + backward + optimize
            h = self.encoder(obs_flatten)
            p = self.decoder_prob(h)

            if not isinstance(self.prob_loss, nn.CrossEntropyLoss):
               p = F.softmax(p, dim=1)

            # Get loss function
            val_loss = self.prob_loss(p, gt_prob)

            total_val_loss += val_loss.item()


            #Get accuracy
            acc += (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
            precision_at_4 += get_precision_at_k(5, p, gt_prob)
            precision_at_3 += get_precision_at_k(3, p, gt_prob)
            precision_at_2 += get_precision_at_k(2, p, gt_prob)

            pt_taken, pt_gt = torch.argmax(p, dim=1), torch.argmax(gt_prob, dim=1)
            pt_pos, gt_pos =  obs[torch.arange(obs.shape[0]), pt_taken], obs[torch.arange(obs.shape[0]), pt_gt]
            distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b

         self.prob_scheduler.step(total_val_loss)

      self.val_step = {'val_loss': total_val_loss/ len(val_loader),
                  'accuracy' : acc/len(val_loader),
                  'precision4': precision_at_4 / len(val_loader),
                  'precision3': precision_at_3 / len(val_loader),
                  'precision2': precision_at_2 / len(val_loader),
                  }
      

      # Save Best Model
      if total_val_loss/ len(val_loader) < self.best_prob_network_loss:
         self.best_prob_network_loss = total_val_loss/ len(val_loader)
         
         self.best_model['encoder'] = self.encoder.state_dict()
         self.best_model['decoder_prob'] = self.decoder_prob.state_dict()
         self.best_model['prob_network_loss'] = self.best_prob_network_loss
      return self.val_step


   def test_network(self, test_loader, verbose=False) :

      distance = 0.0
      acc_p, precision_at_4, precision_at_3, precision_at_2 = 0.0, 0.0, 0.0, 0.0

      self.eval()
      with torch.no_grad():
         for i , data in enumerate(test_loader) :

            # get the inputs and labels from the data loader
            obs, gt = data

            b = obs.shape[0]
            obs_flatten = obs.reshape(b, -1)
            gt_prob = gt[0]
            gt_action = gt[1]
            # forward two models
            p = self.forward(obs_flatten)

            #Get accuracy probability
            acc_p += (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b

            #Get precision@k
            precision_at_4 += get_precision_at_k(5, p, gt)
            precision_at_3 += get_precision_at_k(3, p, gt)
            precision_at_2 += get_precision_at_k(2, p, gt)

            if verbose:
               #print(f'Preidiction {i}/ {len(test_loader)}')
               #print('Ground truth     : ', gt[:, -2:].tolist())
               #print('Predicted prob   : ', p.tolist())
               #print('Ground truth prob: ', gt[:, :-2].tolist())


               # Plot on 2d the observation and the prediction
               # obs are [x,y,z] coordinates, while prediction is the point take

               plt.figure(figsize=(10,10))
               obs_plot_pt = obs.reshape(-1, 3)[:,:2].cpu().numpy()
               plt.scatter(obs_plot_pt[:,0], obs_plot_pt[:,1], label='Observation')



               # select where pt is > 0.7 and where is max

               p = F.softmax(p, dim=1)
               fifth_highest_value = torch.sort(p, descending=True).values[0,4]

               idx = torch.where(p > fifth_highest_value)[1]
               max_id = torch.argmax(p, dim=1)


               # Plot ground truth (max of gt)
               gt_idx = gt_prob.argmax(dim=1)[0]
               plt.scatter(obs_plot_pt[gt_idx,0], obs_plot_pt[gt_idx,1], label='GT', s=120, marker='x',color='green')

               
               #pt = obs.reshape(-1,3)[idx, :2].cpu().numpy()
               #plt.scatter(pt[:,0], pt[:,1], label='Prediction', color='red')
               
               prob = p[:,idx].cpu().numpy()

               for j, id in enumerate(idx):
                  # Plot all prediction with scatter
                  plt.scatter(obs_plot_pt[id,0], obs_plot_pt[id,1], label=f'Prediction {id}', color='red')
                  
                  label = prob[0,j]
                  pt = obs_plot_pt[id,:]
                  if id == max_id:
                     plt.annotate(f'{label:.2f}', (pt[0], pt[1]), textcoords="offset points", xytext=(0, 10), ha='center', weight='bold', fontsize=12)
                  else:
                     plt.annotate(f'{label:.2f}', (pt[0], pt[1]), textcoords="offset points", xytext=(0, 10), ha='center',fontsize=12)

               base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'img')
               print('Save figure in ', os.path.join(base_path, f'prediction_{i}.png'))
               plt.savefig(os.path.join(base_path, f'prediction_{i}.png'))
               plt.close()
               plt.clf()
               #print('\n\n')


         self.test_step = {'distance' : distance / len(test_loader),
                           'accuracy_prob' : acc_p / len(test_loader),
                           'precision4': precision_at_4 / len(test_loader),
                           'precision3': precision_at_3 / len(test_loader),
                           'precision2': precision_at_2 / len(test_loader),
                           }
   
         return self.test_step




   def set_prob_model(self):
      self.encoder.load_state_dict(self.best_model['encoder'])
      self.decoder_prob.load_state_dict(self.best_model['decoder_prob'])


   def set_best_model(self, encoder =None, decoder_prob = None):
      
      if encoder is not None:
         self.encoder.load_state_dict(encoder)
      else :
         self.encoder.load_state_dict(self.best_model['encoder'])

      if decoder_prob is not None:
         self.decoder_prob.load_state_dict(decoder_prob)
      else :
         self.decoder_prob.load_state_dict(self.best_model['decoder_prob'])


   def load_model(self, encoder, decoder_prob) :
      self.encoder.load_state_dict(encoder)
      self.decoder_prob.load_state_dict(decoder_prob)


   def save_best_model(self, path):

      data = {'init_params': self.init_params,
               'encoder': self.best_model['encoder'],
               'decoder_prob': self.best_model['decoder_prob'],
               'prob_network_loss': self.best_prob_network_loss,
      }

      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)

      torch.save(data, path)




if __name__ == '__main__' :

   p = ProbNetwork(no_points=400, action_dim=2, hidden_dim=128)


   input_t = torch.randn(1200).to(p.device)

   output = p(input_t)