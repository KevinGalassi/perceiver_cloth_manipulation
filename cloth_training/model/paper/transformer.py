import torch
import torch.nn as nn


from cloth_training.model.common.model_utils import set_seed
from cloth_training.model.common.model_utils import Lamb

from cloth_training.model.common.attention_models import Attention, FeedForward
from einops import repeat


import os



class DaggerTransformer(nn.Module):

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

      input_embedding_dim   = kwargs.get('input_embedding_dim')
      num_latent_heads      = kwargs.get('num_latent_heads')
      num_output_heads      = kwargs.get('num_output_heads')
      self.depth  = kwargs.get('depth')
      self.lr          = kwargs.get('lr')
      seed        = kwargs.get('seed')

      set_seed(seed)

      self.init_params = dict(kwargs)



      # Input Feature Embedding
      self.input_embedding = nn.Sequential(
         nn.Linear(3, input_embedding_dim//2),
         nn.ReLU(),
         nn.Linear(input_embedding_dim//2, input_embedding_dim)
      )

      self.transformer_layers = nn.ModuleList()
      for _ in range(self.depth):
         self.transformer_layers.append(Attention(embed_dim=input_embedding_dim, num_heads=num_latent_heads, batch_first=True))
         self.transformer_layers.append(FeedForward(input_embedding_dim))


      #Output latent vector
      self.output = nn.Parameter(torch.normal(0, 0.2, (1, input_embedding_dim)))

      # Output cross-attention of latent space with input as query      
      self.output_cross_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_output_heads, batch_first=True)
      
      # Decoder
      self.output_layer = nn.Sequential(
               nn.Linear(input_embedding_dim, input_embedding_dim//2),
               nn.ReLU(),
               nn.Linear(input_embedding_dim//2, 4)
            )

      self.criterion  = nn.MSELoss()

      self.best_val_loss = float('inf')


   def reset_train(self) :
      self.criterion = nn.MSELoss()
      self.optimizer = Lamb(self.parameters(), lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')



   def forward(self, x):
      x = self.input_embedding(x)

      for i in range(0, len(self.transformer_layers), 2):
         x = self.transformer_layers[i](x)
         x = self.transformer_layers[i+1](x) + x


      o = repeat(self.output, 'n d -> b n d', b = x.shape[0])
      o = self.output_cross_attention(o, context = x)
      x = self.output_layer(o)

      return x.tanh()


   def trainer(self, train_loader, device='cuda') :

      self.train()
      total_train_loss = 0.0

      distance_prob, distance_d, distance_dx, distance_dy, angle = 0.0, 0.0, 0.0, 0.0, 0.0

      for i, data in enumerate(train_loader) :
      #for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Progress'):
         # get the inputs and labels from the data loader
         obs, gt = data

         b = obs[0].shape[0]

         pts = obs[0].to(device)
         gt_gaussian = gt[0].to(device)
         gt_action = gt[1].to(device)

         p = self.forward(pts)
         p = p.reshape(b,-1)
         # Get loss function

         loss = self.criterion(p, gt_action)

         self.optimizer.zero_grad()
         loss.backward()
         self.optimizer.step()

         total_train_loss += loss.item()

         lr = self.optimizer.param_groups[0]["lr"]

         distance_prob += torch.sqrt(torch.sum((p[:,:2] - gt_action[:,:2])**2, dim=1)).sum().item() / b
         distance_d += torch.sqrt(torch.sum((p[:,2:] - gt_action[:,2:])**2, dim=1)).sum().item() / b
         distance_dx += torch.sqrt(torch.sum((p[:,-2] - gt_action[:,-2])**2, dim=0)).sum().item() / b
         distance_dy += torch.sqrt(torch.sum((p[:,-1] - gt_action[:,-1])**2, dim=0)).sum().item() / b
         angle += torch.sqrt(torch.sum((torch.atan2(p[:,-1], p[:,-2]) - torch.atan2(gt_action[:,-1], gt_action[:,-2]))**2, dim=0)).sum().item() / b
                         
      self.training_step = {'train_loss': total_train_loss / len(train_loader),
                           'distance_d': distance_d / len(train_loader),
                           'distance_dx': distance_dx / len(train_loader),
                           'distance_dy': distance_dy / len(train_loader),
                           'angle': angle / len(train_loader),
                           'distance_prob': distance_prob / len(train_loader),
                           'lr': lr
                           }

      return self.training_step
   

   def validate(self, val_loader, device='cuda') :
      # validate the model
      total_val_loss = 0.0
      distance_prob, distance_d, distance_dx, distance_dy, angle = 0.0, 0.0, 0.0, 0.0, 0.0

      self.eval()
      with torch.no_grad():

         val_step = {}
         for i , data in enumerate(val_loader) :

            obs, gt = data

            b = obs[0].shape[0]

            pts = obs[0].to(device)
            gt_gaussian = gt[0].to(device)
            gt_action = gt[1].to(device)

            p = self.forward(pts)
            p = p.reshape(b,-1)

            # Get loss function
            prob_loss = self.criterion(p, gt_action)
            
            total_val_loss += prob_loss.item()

            #Get accuracy
            distance_prob += torch.sqrt(torch.sum((p[:,:2] - gt_action[:,:2])**2, dim=1)).sum().item() / b
            distance_d += torch.sqrt(torch.sum((p[:,2:] - gt_action[:,2:])**2, dim=1)).sum().item() / b
            distance_dx += torch.sqrt(torch.sum((p[:,-2] - gt_action[:,-2])**2, dim=0)).sum().item() / b
            distance_dy += torch.sqrt(torch.sum((p[:,-1] - gt_action[:,-1])**2, dim=0)).sum().item() / b
            angle += torch.sqrt(torch.sum((torch.atan2(p[:,-1], p[:,-2]) - torch.atan2(gt_action[:,-1], gt_action[:,-2]))**2, dim=0)).sum().item() / b


      self.scheduler.step(total_val_loss)
      val_step = {'val_loss': total_val_loss/ len(val_loader),
                        'distance_d': distance_d / len(val_loader),
                        'distance_dx': distance_dx / len(val_loader),
                        'distance_dy': distance_dy / len(val_loader),
                        'angle': angle / len(val_loader),
                        'distance_prob': distance_prob / len(val_loader),
                        }
      
      # Save Best Model
      if total_val_loss/ len(val_loader) < self.best_val_loss:
         self.best_val_loss = total_val_loss/ len(val_loader)

      return val_step

   def save_best_model(self, path):
      state = {'model': self.state_dict(),
               'init_params': self.init_params,
               }
      
      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)

      torch.save(state, path)

   def get_best_result(self):
      return {'best_train_loss': self.best_train_loss, 'best_val_loss': self.best_val_loss}