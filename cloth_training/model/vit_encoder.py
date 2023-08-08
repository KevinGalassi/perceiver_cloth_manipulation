import torch
import torch.nn as nn
import numpy as np
from cloth_training.model.model_architecture.dataset_gen import GymClothDataset
from cloth_training.model.model_architecture.attention_models import Attention, FeedForward
from cloth_training.model.model_architecture.model_utils import Lamb


from einops import repeat

import os



def set_seed(seed):
   if seed == None :
      print('Warning : seed is None')
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True


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




class VisionTransformer(nn.Module):
   def __init__(self, **kwargs) : 
      super(VisionTransformer, self).__init__()


      depth = kwargs.get('depth', 1)
      image_size  = kwargs.get('image_size', 224)
      patch_size  = kwargs.get('patch_size', 16)
      num_points     = kwargs.get('num_points', 625)
      num_cross_heads = kwargs.get('num_cross_heads', 8)
      num_vit_heads  = kwargs.get('num_vit_heads', 8)
      self.loss_type = kwargs.get('loss_type', 'chamfer') 
      self.lr = kwargs.get('lr', 1e-3)
      seed = kwargs.get('seed', None)

      self.init_params = dict(kwargs)

      # Calculate the number of patches and the dimension of each patch
      num_patches = (image_size // patch_size) ** 2
      patch_dim = 3 * patch_size ** 2  # Assuming input image has 3 channels (RGB)
      c_out = patch_size * patch_size * 3


      # Positional embeddings
      self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, patch_dim))

      # Patch embeddings
      self.patch_embeddings = nn.Conv2d(3, patch_dim, kernel_size=patch_size, stride=patch_size)

      # Transformer layers
      self.transformer_layers = nn.ModuleList([
         nn.TransformerEncoderLayer(d_model=patch_dim, nhead=num_vit_heads)
         for _ in range(depth)  # You can adjust the number of transformer layers as needed
      ])

      # Cross-attention module
      self.latents = nn.Parameter(torch.normal(0, 0.2, (num_points, c_out)))

      self.cross_attention = Attention(embed_dim=c_out, num_heads=num_cross_heads, batch_first=True)
      self.cross_ff = FeedForward(c_out)
      
      self.output_layer = nn.Sequential(
               nn.Linear(c_out, c_out//2),
               nn.ReLU(),
               nn.Linear(c_out//2, 3)
            )
      
      # Final classification head
      self.fc = nn.Linear(patch_dim, num_points)

   def forward(self, x):
      x = self.patch_embeddings(x)  # Convert input image to patches
      x = x.flatten(2).transpose(1, 2)  # Flatten and transpose to B x (N+1) x C (N: number of patches)
      x = torch.cat((torch.zeros(x.size(0), 1, x.size(2)).to(x.device), x), dim=1)  # Add the learnable positional embedding
      x = x + self.position_embeddings  # Add the positional embeddings to the patches

      for layer in self.transformer_layers:
         x = layer(x)

      # Cross-attention   
      o = repeat(self.latents, 'n d -> b n d', b=x.size(0))

      x = self.cross_attention(o, x)
      x = self.cross_ff(x)

      x = self.output_layer(x)

      return x
   
    
   def reset_train(self) :
      if self.loss_type == 'chamfer' :
         self.criterion = ChamferLoss()
      else :
         self.criterion = nn.MSELoss()
      self.optimizer = Lamb(self.parameters(), lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')


   
   def trainer(self, train_loader, device = 'cuda') :

      self.train()
      total_train_loss = 0.0

      for i, data in enumerate(train_loader) :
      #for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Progress'):
         # get the inputs and labels from the data loader
         obs, gt = data

         rgb = obs.to(device).float()
         gt_pts = gt.to(device)

         b = rgb.shape[0]
        

         # forward + backward + optimize
         p = self.forward(rgb)
        
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

            rgb = obs.to(device).float()
            gt_pts = gt.to(device)

            b = rgb.shape[0]

            # forward + backward + optimize
            p = self.forward(rgb)
         
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
      os.makedirs(folder_path, exist_ok=True)

      torch.save(state, path)


if __name__ == '__main__' :
   

   hparams = {
               'depth' : 1,   
               'num_points' : 625,
               'patch_size' : 16,
               'image_size' : 224,
               'num_cross_heads' : 8,
               'num_vit_heads' : 8,
               'lr' : 0.001,
               'seed' : 42
            }
   

   agent = VisionTransformer(**hparams)


   dataset = torch.load('/home/kgalassi/code/cloth/cloth_training/cloth_training/dagger/pull/pull_dataset.pt')
   dataset.set_obs_type('vit')
   dataset.set_output_type('vit')
   dataset.to_device('cpu')

   obs, gt = dataset[0]

   print(obs.shape)

   train_loader = []
   for _ in range(5):
      obs = torch.rand(10,3, 224,224)
      gt = torch.rand(10, 625,3)

      train_loader.append((obs, gt))


   agent.to('cuda')
   

   agent.reset_train()
   agent.trainer(train_loader)
   agent.validate(train_loader)
