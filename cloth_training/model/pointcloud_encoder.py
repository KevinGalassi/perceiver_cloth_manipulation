import torch
import torch.nn as nn
import numpy as np
from cloth_training.dataset.dataset_gen import GymClothDataset
from cloth_training.model.common.attention_models import Attention, FeedForward
from cloth_training.model.common.model_utils import Lamb


from einops import repeat

import os

from cloth_training.model.common.model_utils import set_seed


class ChamferLoss(nn.Module):
   def __init__(self):
      super(ChamferLoss, self).__init__()

   def forward(self, x, y):
      x = x.unsqueeze(1)  # (B, 1, N, D)
      y = y.unsqueeze(2)  # (B, M, 1, D)
      dist = torch.norm(x - y, dim=-1)  # (B, M, N)
      dist1, _ = torch.min(dist, dim=-1)  # (B, M)
      dist2, _ = torch.min(dist, dim=-2)  # (B, N)
      return torch.mean(dist1) + torch.mean(dist2)



class PointCloudEncoder(nn.Module) :
   def __init__(self, **kwargs) -> None :


      self.depth           = kwargs.get('depth')
      input_latent_dim     = kwargs.get('input_latent_dim', 625)
      input_embedding_dim  = kwargs.get('input_embedding_dim')
      num_latent_heads     = kwargs.get('num_latent_heads')
      num_cross_heads      = kwargs.get('num_cross_heads')
      self.num_latent_layers = kwargs.get('num_latent_layers')
      self.loss_type = kwargs.get('loss_type', 'chamfer') 

      self.lr              = kwargs.get('lr')
      seed                 = kwargs.get('seed')

      super().__init__()

      self.init_params = dict(kwargs)
      set_seed(seed)

      self.latents = nn.Parameter(torch.normal(0, 0.2, (input_latent_dim, input_embedding_dim)))


      self.ptc_embedd = nn.Sequential(
               nn.Linear(3, input_embedding_dim//2),
               nn.ReLU(),
               nn.Linear(input_embedding_dim//2, input_embedding_dim),
            )
      
      # Latent trasnformer module
      self.self_latent_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_latent_heads, batch_first=True)
      self.self_latent_ff = FeedForward(input_embedding_dim)

      # Cross-attention module
      self.cross_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_cross_heads, batch_first=True)
      self.cross_ff = FeedForward(input_embedding_dim)
      
      self.output_layer = nn.Sequential(
               nn.Linear(input_embedding_dim, input_embedding_dim//2),
               nn.ReLU(),
               nn.Linear(input_embedding_dim//2, 3)
            )

   def reset_train(self) :
      if self.loss_type == 'chamfer' :
         self.criterion = ChamferLoss()
      else :
         self.criterion = nn.MSELoss()
      self.optimizer = Lamb(self.parameters(), lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')


   def forward(self, ptc) :

      batch = ptc.shape[0]

      ptce = self.ptc_embedd(ptc)
      x = repeat(self.latents, 'n d -> b n d', b=batch)

      for _ in range(self.depth):
         x = self.cross_attention(x, context = ptce)
         x = self.cross_ff(x) + x

         for _ in range(self.num_latent_layers):
            x = self.self_latent_attention(x) 
            x = self.self_latent_ff(x) + x

      p = self.output_layer(x)

      return p


   
   def trainer(self, train_loader, device = 'cuda') :

      self.train()
      total_train_loss = 0.0

      for i, data in enumerate(train_loader) :
      #for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Progress'):
         # get the inputs and labels from the data loader
         obs, gt = data

         ptc = obs.to(device)
         gt_pts = gt.to(device)

         b = ptc.shape[0]
        

         # forward + backward + optimize
         p = self.forward(ptc)
        
         # Get loss function
         loss = self.criterion(p, gt_pts)
         


         # zero the parameter gradients
         self.optimizer.zero_grad()
         loss.backward()
         #torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
         self.optimizer.step()

         total_train_loss += loss.item()

         # Log learning rate to Tensorboard
         lr = self.optimizer.param_groups[0]["lr"]
                         
      self.training_step = {'train_loss': total_train_loss / len(train_loader),
                           'lr': lr
                           }

      return self.training_step
   
   def validate(self, val_loader, device='cuda') :
      # validate the model
      total_val_loss = 0.0
      best_val_loss = float('inf')
      self.eval()
      with torch.no_grad():

         val_step = {}
         for i , data in enumerate(val_loader) :

            obs, gt = data

            ptc = obs.to(device)
            gt_pts = gt.to(device)

            b = ptc.shape[0]

            

            # forward + backward + optimize
            p = self.forward(ptc)
         
            # Get loss function
            loss = self.criterion(p, gt_pts)

            
            total_val_loss += loss.item()



      self.scheduler.step(total_val_loss)
      val_step = {'val_loss': total_val_loss/ len(val_loader),
                  }
      
      # Save Best Model
      if total_val_loss/ len(val_loader) < best_val_loss:
         self.best_train_loss = val_step['val_loss']
         best_val_loss = total_val_loss/ len(val_loader)
         self.best_model = self.state_dict()

         
      return val_step
   
   def save_best_model(self, path):

      state = {'init_params': self.init_params,
               'model': self.state_dict(),
      }

      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      print(folder_path)
      os.makedirs(folder_path, exist_ok=True)

      torch.save(state, path)


if __name__ == '__main__' :
   

   hparams = {
               'depth' : 2,
               'input_latent_dim' : 625,
               'input_embedding_dim' : 128,
               'num_latent_heads' : 8,
               'num_cross_heads' : 8,
               'num_latent_layers' : 2,
               'lr' : 0.001,
               'seed' : 42
            }
   

   agent = PointCloudEncoder(**hparams)

   train_loader = []
   for _ in range(5):
      random_number = torch.randint(low=0, high=2561, size=(1,)).item()
      obs = torch.rand(10,random_number,3)
      gt = torch.rand(10, 625,3)

      train_loader.append((obs, gt))


   agent.to('cuda')
   agent.reset_train()


   agent.trainer(train_loader)
   agent.validate(train_loader)
