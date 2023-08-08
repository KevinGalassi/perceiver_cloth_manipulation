import torch
import torch.nn as nn
import torch.optim as optim

class ImageNetwork(nn.Module):

   def __init__(self, **kwargs):
      super(ImageNetwork, self).__init__()
      
      self.lr = kwargs.get('lr', 10e-4)

      self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
      self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
      self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
      self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
      
      self.dense1 = nn.Linear(32 * 224 * 224, 256)
      self.dense2 = nn.Linear(256, 256)
      self.dense3 = nn.Linear(256, 256)
      self.dense4 = nn.Linear(256, 4)

      self.relu = nn.ReLU()

      self.best_model = {}
      self.init_params = kwargs

      prob_params = list(self.parameters())
      self.prob_optimizer = optim.Adam(prob_params, lr=self.lr, weight_decay=10e-5)
      self.prob_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.prob_optimizer, 'min')
      self.mse_loss = nn.MSELoss()
      self.best_prob_network_loss = float('inf')


   def forward(self, input):

      x = input.float()
      x = x.permute(0, 3, 1, 2)

      x = self.conv1(x)
      x = self.relu(x)
      x = self.conv2(x)
      x = self.relu(x)
      x = self.conv3(x)
      x = self.relu(x)
      x = self.conv4(x)
      x = self.relu(x)

      x = x.reshape(-1, 32 * 224 * 224)

      x = self.dense1(x)
      x = self.relu(x)
      x = self.dense2(x)
      x = self.relu(x)
      x = self.dense3(x)
      x = self.relu(x)
      x = self.dense4(x)
      

      return nn.Tanh()(x)



   def trainer(self, train_loader, device = torch.device('cuda')):
      self.train()
      total_train_loss = 0.0
      total_train_prob_loss = 0.0
      total_train_action_loss = 0.0

      distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0
      distance_prob = 0.0

      for i , data in enumerate(train_loader) :
         # get the inputs and labels from the data loader
         obs, gt = data


         b = obs.shape[0]

         obs = obs.to(device)
         gt_prob = gt[0].to(device).float()
         gt_action = gt[1].to(device).float()

         # forward encoder and probability network
         a = self.forward(obs)

         loss = self.mse_loss(a, torch.cat([gt_prob, gt_action], dim=-1))
       
         #Get accuracy
         pt_pos, gt_pos =  a[:,:2], gt_prob
         distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b


         # Distance
         distance += torch.sqrt(torch.sum((a[:,2:] - gt_action)**2, dim=1)).sum().item() / b
         distance_x += torch.sqrt(torch.sum((a[:,-2] - gt_action[:, -2])**2, dim=0)).sum().item() / b
         distance_y += torch.sqrt(torch.sum((a[:,-1] - gt_action[:, -1])**2, dim=0)).sum().item() / b
         angle += torch.sqrt(torch.sum((torch.atan2(a[:,-1], a[:,-2]) - torch.atan2(gt_action[:, -1], gt_action[:, -2]))**2, dim=0)).sum().item() / b


         # zero the parameter gradients
         self.prob_optimizer.zero_grad()
         loss.backward()
         self.prob_optimizer.step()

         total_train_loss += loss.item()

         # Log learning rate to Tensorboard
      lr = self.prob_optimizer.param_groups[0]["lr"]

      self.training_step = {'train_loss': total_train_loss / len(train_loader),
                           'train_prob_loss': total_train_prob_loss / len(train_loader),
                           'train_action_loss': total_train_action_loss / len(train_loader),
                           'distance_prob': distance_prob / len(train_loader),
                           'distance': distance / len(train_loader),
                           'distance_x': distance_x / len(train_loader),
                           'distance_y': distance_y / len(train_loader),
                           'angle': angle / len(train_loader),
                           'lr': lr}
      return self.training_step


   def validate(self, val_loader, device = torch.device('cuda')):

      total_val_loss = 0.0
      distance_prob = 0.0
      distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0

      self.eval()
      with torch.no_grad():
         for i, data in enumerate(val_loader) :
            # get the inputs and labels from the data loader
            obs, gt = data


            b = obs.shape[0]

            obs = obs.to(device)
            gt_prob = gt[0].to(device).float()
            gt_action = gt[1].to(device).float()

            # forward encoder and probability network
            a = self.forward(obs)

            loss = self.mse_loss(a, torch.cat([gt_prob, gt_action], dim=-1))
         
            #Get accuracy
            pt_pos, gt_pos =  a[:,:2], gt_prob
            distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b


            # Distance
            distance += torch.sqrt(torch.sum((a[:,2:] - gt_action)**2, dim=1)).sum().item() / b
            distance_x += torch.sqrt(torch.sum((a[:,-2] - gt_action[:, -2])**2, dim=0)).sum().item() / b
            distance_y += torch.sqrt(torch.sum((a[:,-1] - gt_action[:, -1])**2, dim=0)).sum().item() / b
            angle += torch.sqrt(torch.sum((torch.atan2(a[:,-1], a[:,-2]) - torch.atan2(gt_action[:, -1], gt_action[:, -2]))**2, dim=0)).sum().item() / b

            total_val_loss += loss.item()



         self.prob_scheduler.step(total_val_loss)

      if len(val_loader) == 0:
         return {}
      
      self.val_step = {'train_loss': total_val_loss / len(val_loader),
                        'distance_prob': distance_prob / len(val_loader),
                        'distance': distance / len(val_loader),
                        'distance_x': distance_x / len(val_loader),
                        'distance_y': distance_y / len(val_loader),
                        'angle': angle / len(val_loader),
                        }

      # Save Best Model
      if total_val_loss/ len(val_loader) < self.best_prob_network_loss:
         self.best_prob_network_loss = total_val_loss/ len(val_loader)
         self.best_model['model'] = self.state_dict()

      return self.val_step


   def reset_training(self):
      prob_params = list(self.parameters())
      self.best_prob_network_loss = 100000
      self.prob_optimizer = optim.Adam(prob_params, lr=self.lr, weight_decay=1e-5)
      self.prob_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.prob_optimizer, 'min')
   

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
   


import os, time, pickle
from tqdm import tqdm
import numpy as np
from cloth_training.model.model_architecture.dataset_gen import GymClothDataset


from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

   save_result_to = 'image_net'
   dataset_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')

   hparams = {
         'learning_rate': 0.001,
         'num_epochs': 150,
         'batch_size': 128,
         'seed' : 0,
         'val_ratio' : 0.2,
   }


   dataset = torch.load('/home/kgalassi/code/cloth/cloth_training/cloth_training/dataset/pull/pulldataset.pt')
   dataset.set_obs_type('rgb')
   dataset.set_output_type('offset')
   dataset.to_device(torch.device('cuda'))



   model = torch.load('/home/kgalassi/gym-cloth-logs/image/dagger-tier-3-pol-pull-image-network/3.pth')

   hparams = model['init_params']
   weight = model['model']

   # Define agent and model
   agent = ImageNetwork(**hparams)

   agent.load_state_dict(weight)

   agent.to(torch.device('cuda'))


   log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', save_result_to)
   model_base_path   = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model', save_result_to)
   run_id = time.strftime("%m-%d-%H-%M")

   writer = SummaryWriter(log_dir=os.path.join(log_path, str(run_id)))
   writer.add_text('Hyperparameters', str(hparams))
   writer.flush()
   


   if 'test_ratio' in hparams :
      test_ratio = hparams['test_ratio']
   else :
      test_ratio = 0.05

   num_val_samples = int(len(dataset) * hparams['val_ratio'])
   num_train_samples = len(dataset) - num_val_samples

   train_d, val_d = torch.utils.data.random_split(dataset, [num_train_samples, num_val_samples])

   train_loader = DataLoader(train_d, batch_size=hparams['batch_size'], shuffle=True)
   val_loader   = DataLoader(val_d, batch_size=hparams['batch_size'], shuffle=False)



   print(f'Logging results to : {save_result_to}')
   for epoch in tqdm(range(hparams['num_epochs']), desc='Epoch training'):

      epoch_train_result = agent.trainer(train_loader)
      for key, value in epoch_train_result.items():
         writer.add_scalar(f'Train/{key}', value, epoch)
      writer.flush()

      epoch_val_result = agent.validate(val_loader)

      for key, value in epoch_val_result.items():
         writer.add_scalar(f'Val/{key}', value, epoch)
      writer.flush()

      model_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_model', save_result_to)
      model_path = os.path.join(model_base_path, str(run_id)+'.pth')
      agent.save_best_model(model_path)
         
      with open(os.path.join(model_base_path, f'{run_id}_hparams.pickle'), 'wb') as f:
         pickle.dump(hparams, f, protocol=4)
      





