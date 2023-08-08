import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cloth_training.model.model_architecture.model_utils import get_precision_at_k
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

from cloth_training.model.model_architecture.base_model import OneStageNetwork

import matplotlib.pyplot as plt

class JointNetwork(nn.Module):

   def __init__(self, **kwargs) -> None:
      '''
      Kwargs list :
      - no_points : number of points in the cloth (default: 400)
      - action_dim : number of action dimension _offset_ (default: 2)
      - hidden_dim : number of hidden dimension (default: 128)
      '''

      super().__init__()

      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      no_points         = int(kwargs.get('no_points', 400))
      action_dim        = int(kwargs.get('action_dim', 2))
      hidden_dim        = int(kwargs.get('hidden_dim', 128))
      self.lr           = kwargs.get('lr', 1e-3)

      if kwargs.get('seed', None) is not None:
         set_seed(kwargs.get('seed'))

      # copy kwargs to init_params
      self.init_params = kwargs.copy()

      #self.init_params = {'no_points': no_points,
      #                     'action_dim': action_dim,
      #                     'hidden_dim': hidden_dim,
      #                     'lr': self.lr,
      #                     'seed': kwargs.get('seed', None)
      #                     }
      
      print(no_points*3)
      self.model = OneStageNetwork(no_points*3, hidden_dim, no_points, 2).to(self.device)

      self.prob_loss = nn.CrossEntropyLoss()
      self.mse_loss = nn.MSELoss()


      # Grasp Probability parameters
      prob_params = list(self.model.parameters())
      
      self.prob_optimizer = optim.Adam(prob_params, lr=self.lr)
      self.prob_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.prob_optimizer, 'min')

      self.best_model = {}
      self.best_prob_network_loss = float('inf')
      
   
   def forward(self, x) :

      p, a = self.model(x)

      return p, a



   def trainer(self, train_loader):
      self.train()
      total_train_loss = 0.0
      total_train_prob_loss = 0.0
      total_train_action_loss = 0.0


      acc, precision_at_4, precision_at_3,precision_at_2, distance_prob = 0.0, 0.0, 0.0, 0.0, 0.0
      distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0
  

      for i , data in enumerate(train_loader) :
         # get the inputs and labels from the data loader
         obs, gt = data

         b = obs.shape[0]
         obs_flatten = obs.reshape(b, -1)

         gt_prob, gt_action = gt[0], gt[1]

         # forward encoder and probability network
         p, a = self.model(obs_flatten)

         p_loss = self.prob_loss(p, gt_prob)
         a_loss = self.mse_loss(a, gt_action)

         loss = p_loss + a_loss

         



         #Get accuracy
         acc += (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
         precision_at_4 += get_precision_at_k(5, p, gt_prob)
         precision_at_3 += get_precision_at_k(3, p, gt_prob)
         precision_at_2 += get_precision_at_k(2, p, gt_prob)

         pt_taken, pt_gt = torch.argmax(p, dim=1), torch.argmax(gt_prob, dim=1)
         pt_pos, gt_pos =  obs[torch.arange(b), pt_taken], obs[torch.arange(b), pt_gt]
         distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b


         # Distance
         distance += torch.sqrt(torch.sum((a - gt_action)**2, dim=1)).sum().item() / b
         distance_x += torch.sqrt(torch.sum((a[:,0] - gt_action[:, -2])**2, dim=0)).sum().item() / b
         distance_y += torch.sqrt(torch.sum((a[:,1] - gt_action[:, -1])**2, dim=0)).sum().item() / b
         angle += torch.sqrt(torch.sum((torch.atan2(a[:,1], a[:,0]) - torch.atan2(gt_action[:, -1], gt_action[:, -2]))**2, dim=0)).sum().item() / b


         # zero the parameter gradients
         self.prob_optimizer.zero_grad()
         loss.backward()
         self.prob_optimizer.step()

         total_train_loss += loss.item()
         total_train_prob_loss += p_loss.item()
         total_train_action_loss += a_loss.item()

         # Log learning rate to Tensorboard
      lr = self.prob_optimizer.param_groups[0]["lr"]

      self.training_step = {'train_loss': total_train_loss / len(train_loader),
                           'train_prob_loss': total_train_prob_loss / len(train_loader),
                           'train_action_loss': total_train_action_loss / len(train_loader),
                           'accuracy' : acc / len(train_loader),
                           'precision4': precision_at_4 / len(train_loader),
                           'precision3': precision_at_3 / len(train_loader),
                           'precision2': precision_at_2 / len(train_loader),
                           'distance_prob': distance_prob / len(train_loader),
                           'distance': distance / len(train_loader),
                           'distance_x': distance_x / len(train_loader),
                           'distance_y': distance_y / len(train_loader),
                           'angle': angle / len(train_loader),
                           'lr': lr}
      return self.training_step


   def validate(self, val_loader):

      total_val_loss, total_val_prob_loss, total_val_action_loss = 0.0, 0.0, 0.0
      acc, precision_at_4, precision_at_3, precision_at_2, distance_prob = 0.0, 0.0, 0.0, 0.0, 0.0
      distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0

      self.eval()
      with torch.no_grad():
         for i, data in enumerate(val_loader) :
            # get the inputs and labels from the data loader
            obs, gt = data

            b = obs.shape[0]
            obs_flatten = obs.reshape(b, -1)
            gt_prob = gt[0]
            gt_action = gt[1]


            # forward + backward + optimize
            p, a = self.model(obs_flatten)


            p_loss = self.prob_loss(p, gt_prob)
            a_loss = self.mse_loss(a, gt_action)

            loss = p_loss + a_loss


            # Get loss function

            total_val_loss += loss.item()
            total_val_prob_loss += p_loss.item()
            total_val_action_loss += a_loss.item()

            #Get accuracy
            acc += (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
            precision_at_4 += get_precision_at_k(5, p, gt_prob)
            precision_at_3 += get_precision_at_k(3, p, gt_prob)
            precision_at_2 += get_precision_at_k(2, p, gt_prob)

            pt_taken, pt_gt = torch.argmax(p, dim=1), torch.argmax(gt_prob, dim=1)
            pt_pos, gt_pos =  obs[torch.arange(obs.shape[0]), pt_taken], obs[torch.arange(obs.shape[0]), pt_gt]
            distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b

            # Distance
            distance += torch.sqrt(torch.sum((a - gt_action)**2, dim=1)).sum().item() / b
            distance_x += torch.sqrt(torch.sum((a[:,0] - gt_action[:, -2])**2, dim=0)).sum().item() / b
            distance_y += torch.sqrt(torch.sum((a[:,1] - gt_action[:, -1])**2, dim=0)).sum().item() / b
            angle += torch.sqrt(torch.sum((torch.atan2(a[:,1], a[:,0]) - torch.atan2(gt_action[:, -1], gt_action[:, -2]))**2, dim=0)).sum().item() / b



         self.prob_scheduler.step(total_val_loss)

      if len(val_loader) == 0:
         return {}
      
      self.val_step = {'train_loss': total_val_loss / len(val_loader),
                           'train_prob_loss': total_val_prob_loss / len(val_loader),
                           'train_action_loss': total_val_action_loss / len(val_loader),
                           'accuracy' : acc / len(val_loader),
                           'precision4': precision_at_4 / len(val_loader),
                           'precision3': precision_at_3 / len(val_loader),
                           'precision2': precision_at_2 / len(val_loader),
                           'distance_prob': distance_prob / len(val_loader),
                           'distance': distance / len(val_loader),
                           'distance_x': distance_x / len(val_loader),
                           'distance_y': distance_y / len(val_loader),
                           'angle': angle / len(val_loader),
                           }

      # Save Best Model
      if total_val_loss/ len(val_loader) < self.best_prob_network_loss:
         self.best_prob_network_loss = total_val_loss/ len(val_loader)
         self.best_model['model'] = self.model.state_dict()

      return self.val_step


   def test_network(self, test_loader, verbose=False) :

      distance = 0.0
      acc_p, precision_at_4, precision_at_3, precision_at_2 = 0.0, 0.0, 0.0, 0.0

      self.eval()
      with torch.no_grad():
         for i , data in enumerate(test_loader) :

            # get the inputs and labels from the data loader
            obs, gt = data

            obs_flatten = obs.reshape(obs.shape[0], -1)
            gt_prob = gt[0]
            gt_action = gt[1]

            # forward two models
            p, a = self.model(obs_flatten)

            #Get accuracy probability
            acc_p += (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b

            #Get precision@k
            precision_at_4 += get_precision_at_k(5, p, gt)
            precision_at_3 += get_precision_at_k(3, p, gt)
            precision_at_2 += get_precision_at_k(2, p, gt)

            if verbose:
               #print(f'Preidiction {i}/ {len(test_loader)}')
               #print('Ground truth     : ', gt_action.tolist())
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

   def reset_training(self):
      self.best_prob_network_loss = 100000
      prob_params = list(self.model.parameters())
      self.prob_optimizer = optim.Adam(prob_params, lr=self.lr)
      self.prob_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.prob_optimizer, 'min')

   def load_model(self, model):
      self.model.load_state_dict(model)




   def save_best_model(self, path):

      data = {'init_params': self.init_params,
                    'model': self.best_model['model'],
      }

      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)

      torch.save(data, path)
   

from cloth_training.train.utils import trainer
from cloth_training.model.model_architecture.dataset_gen import GymClothDataset
if __name__ == '__main__':


   model       = torch.load('./cloth_training/saved_model/single_network/07-07-14-02.pth', map_location=torch.device('cpu'))
   init_params = model['init_params']

   init_params = {'learning_rate': 0.001, 'batch_size': 128, 'num_epochs': 150, 'hidden_dim': 128, 'val_ratio': 0.1, 'test_ratio': 0.05, 'seed': 0}



   hparams = {
         'learning_rate': 0.001,
         'batch_size': 128,
         'num_epochs': 150,
         'hidden_dim':128,
         'val_ratio': 0.10,
         'test_ratio' : 0.05,
         'seed' : 0,
   }



   folder_name = 'single_network'


   #dataset = torch.load('/home/kgalassi/code/cloth/cloth_training/cloth_training/dataset/pull/pulldataset.pt')
   dataset = torch.load('/app/cloth_training/cloth_training/dataset/pull/pulldataset.pt')
   
   dataset.set_obs_type('pts')
   dataset.set_output_type('prob')
   dataset.to_device(torch.device('cuda'))

   num_final_test_sample = int(len(dataset))
   train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_final_test_sample, num_final_test_sample])


   # Define agent and model
   agent = JointNetwork(no_points=625,
                        action_dim = 2,
                        hidden_dim = hparams['hidden_dim'],
                        lr = hparams['learning_rate'])


   save_result_to = 'single_network'
   dataset_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')



   # TRAIN
   trainer(save_result_to, hparams, agent, dataset, save_model=True)


   