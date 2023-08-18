import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from cloth_training.model.model_architecture.model_utils import get_precision_at_k
from cloth_training.model.model_architecture.attention_models import Perceiver
from cloth_training.model.model_architecture.model_utils import Lamb
from cloth_training.model.model_architecture.attention_models import Attention, FeedForward
from einops import repeat
import pickle, os



from cloth_training.model.model_architecture.model_utils import set_seed



class DaggerPerceiver(nn.Module):

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
      num_latents = int(kwargs.get('num_latents', 50))
      num_cross_heads   = int(kwargs.get('num_cross_heads', 8))
      num_output_heads  = int(kwargs.get('num_output_heads', 8))
      num_latent_heads  = int(kwargs.get('num_latent_heads', 8))
      num_latent_layers  = int(kwargs.get('num_latent_layers', 3))
      lr          = kwargs.get('lr', 1e-4)
      seed        = kwargs.get('seed', None)
      self.lr = lr
      

      if seed is not None:
         set_seed(seed)

      self.init_params = {'depth' : depth,
                           'input_dim' : input_dim,
                           'input_embedding_dim' : input_embedding_dim,
                           'num_latents' : num_latents,
                           'num_cross_heads' : num_cross_heads,
                           'num_output_heads' : num_output_heads,
                           'num_latent_heads' : num_latent_heads,
                           'num_latent_layers' : num_latent_layers,
                           'lr' : lr,
                           'seed' : seed,
                        }
      self.num_latent_layers = num_latent_layers
      self.depth = depth
      

      # Input Feature Embedding
      self.input_embedding = nn.Sequential(
         nn.Linear(input_dim, input_embedding_dim),
         nn.ReLU(),
      )

      # Latent Array
      self.latents = nn.Parameter(torch.normal(0, 0.2, (num_latents, input_embedding_dim), device=self.device))

      # Latent trasnformer module
      self.self_latent_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_latent_heads, batch_first=True)
      self.self_latent_ff = FeedForward(input_embedding_dim)

      # Cross-attention module
      self.cross_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_cross_heads, batch_first=True)
      self.cross_ff = FeedForward(input_embedding_dim)
      
      # Output cross-attention of latent space with input as query      
      self.output_cross_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_output_heads, batch_first=True)
      
      # Decoder
      self.output_layer = nn.Sequential(
               nn.Linear(input_embedding_dim, input_embedding_dim//2),
               nn.ReLU(),
               nn.Linear(input_embedding_dim//2, 1)
            )

      self.to(self.device)

      if kwargs.get('output_type', 'prob') == 'prob_gaussian':
         self.criterion = nn.MSELoss()
      elif kwargs.get('output_type', 'prob') == 'prob' :
         self.criterion  = nn.CrossEntropyLoss()
      else :
         raise NotImplementedError

      #self.params = list(self.parameters())
      self.optimizer = Lamb(self.parameters(), lr=lr)
      #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

      #self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

      self.best_model = None
      self.best_val_loss = float('inf')


   def reset_train(self) :
      self.optimizer = Lamb(self.parameters(), lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')



   def forward(self, inputs):
      data = self.input_embedding(inputs)

      b = inputs.shape[0]
      x = repeat(self.latents, 'n d -> b n d', b = b)

      for _ in range(self.depth):
         x = self.cross_attention(x, context = data)
         x = self.cross_ff(x) + x

         for _ in range(self.num_latent_layers):
            x = self.self_latent_attention(x) 
            x = self.self_latent_ff(x) + x

      # Output cross attention
      p = self.output_cross_attention(data, context = x)

      p = self.output_layer(p)
      return p


   

   def trainer(self, train_loader) :

      self.train()
      total_train_loss, total_prob_loss = 0.0, 0.0
      acc, precision_at_4, precision_at_3, precision_at_2, distance_prob = 0.0, 0.0, 0.0, 0.0, 0.0


      for i, data in enumerate(train_loader) :
      #for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Progress'):
         # get the inputs and labels from the data loader
         obs, gt = data

         if obs.isnan().any() or gt[0].isnan().any() or gt[1].isnan().any() :
            print("NAN")

            print(i)

            raise ValueError("NAN")

         b = obs.shape[0]
         gt_prob = gt[0]

         # forward + backward + optimize
         p = self.forward(obs)#.sigmoid()


         p = p.reshape(b, -1)
         # Get loss function
         loss = self.criterion(p, gt_prob)

         if loss.isnan().any() :
            print("NAN")

            print(i)

            raise ValueError("NAN")
         
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
         #torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
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
            gt_prob = gt[0]          
         
            p = self.forward(obs)
            p = p.reshape(b, -1)

            # Get loss function
            prob_loss = self.criterion(p, gt_prob)
            
            total_val_loss += prob_loss.item()

            #Get accuracy
            acc += (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
            precision_at_4 += get_precision_at_k(5, p, gt_prob)
            precision_at_3 += get_precision_at_k(3, p, gt_prob)
            precision_at_2 += get_precision_at_k(2, p, gt_prob)

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
         self.best_model = self.state_dict()

         
      return val_step


   def save_best_model(self, path):

      state = {'init_params': self.init_params,
               'model': self.best_model,
      }

      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      print(folder_path)
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)

      torch.save(state, path)

   def load_model(self, model) :
      self.load_state_dict(model)
      return True

   def get_best_result(self):
      return {'best_train_loss': self.best_train_loss, 'best_val_loss': self.best_val_loss}