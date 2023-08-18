import torch
import torch.nn as nn
import numpy as np

from cloth_training.model.common.model_utils import get_precision_at_k
from cloth_training.model.common.model_utils import Lamb
from cloth_training.model.common.attention_models import Attention, FeedForward
from einops import repeat
import os

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



from cloth_training.model.common.model_utils import set_seed


class HeatPredictor(nn.Module):
   def __init__(self, **kwargs) -> None:
      self.depth            = kwargs.get('depth')
      input_dim             = kwargs.get('input_dim')
      input_embedding_dim   = kwargs.get('input_embedding_dim')
      num_latents           = kwargs.get('num_latents')
      num_cross_heads       = kwargs.get('num_cross_heads')
      num_output_heads      = kwargs.get('num_output_heads')
      num_latent_heads      = kwargs.get('num_latent_heads')
      self.num_latent_layers = kwargs.get('num_latent_layers')
      self.lr                = kwargs.get('lr_heat')
      seed                   = kwargs.get('seed')

      set_seed(seed)
      super().__init__()


      # Input Feature Embedding
      self.input_embedding = nn.Sequential(
         nn.Linear(input_dim, input_embedding_dim),
         nn.ReLU(),
      )

      # Latent Array
      self.latents = nn.Parameter(torch.normal(0, 0.2, (num_latents, input_embedding_dim)))

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
   
   def forward(self, inputs) :
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

   def reset_train(self) :
      self.criterion = nn.MSELoss()
      self.optimizer = Lamb(self.parameters(), lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

   def trainer(self, train_loader, device = 'cuda') :

      self.train()
      total_train_loss = 0.0
      distance_prob = 0.0


      for i, data in enumerate(train_loader) :
      #for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Progress'):
         # get the inputs and labels from the data loader
         obs, gt = data

         pts = obs[0].to(device)
         gt_prob = gt[0].to(device)

         b = pts.shape[0]

         

         # forward + backward + optimize
         p = self.forward(pts)
         p = p.reshape(b, -1)
         
         # Get loss function
         loss = self.criterion(p, gt_prob)
         
         id_taken, id_gt = torch.argmax(p, dim=-1), torch.argmax(gt_prob, dim=-1)
         pt_pos, gt_pos =  pts[torch.arange(b), id_gt], pts[torch.arange(b), id_taken]
         distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=-1)).sum().item() / b


         # zero the parameter gradients
         self.optimizer.zero_grad()
         loss.backward()
         #torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
         self.optimizer.step()

         total_train_loss += loss.item()

         # Log learning rate to Tensorboard
         lr = self.optimizer.param_groups[0]["lr"]
                         
      self.training_step = {'heat_train_loss': total_train_loss / len(train_loader),
                           'distance_prob_heatmap': distance_prob / len(train_loader),
                           'lr': lr
                           }

      return self.training_step
   
   def validate(self, val_loader, device='cuda') :
      # validate the model
      total_val_loss = 0.0
      distance_prob = 0.0
      best_val_loss = float('inf')
      self.eval()
      with torch.no_grad():

         val_step = {}
         for i , data in enumerate(val_loader) :

            obs, gt = data

            pts = obs[0]
            gt_prob = gt[0]          

            #Move to device
            pts = pts.to(device)
            gt_prob = gt_prob.to(device)

            #get batch dimension
            b = pts.shape[0]

            p = self.forward(pts)
            p = p.reshape(b, -1)

            # Get loss function
            prob_loss = self.criterion(p, gt_prob)
            
            total_val_loss += prob_loss.item()

            #Get Distance
            pt_taken, pt_gt = torch.argmax(p, dim=1), torch.argmax(gt_prob, dim=1)
            pt_pos, gt_pos =  pts[torch.arange(pts.shape[0]), pt_taken], pts[torch.arange(pts.shape[0]), pt_gt]
            distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b


      self.scheduler.step(total_val_loss)
      val_step = {'heat_val_loss': total_val_loss/ len(val_loader),
                  'distance_prob_heatmap': distance_prob / len(val_loader),
                  }
      
      # Save Best Model
      if total_val_loss/ len(val_loader) < best_val_loss:
         self.best_train_loss = val_step['val_loss']
         best_val_loss = total_val_loss/ len(val_loader)
         
      return val_step

class PointPredictor(nn.Module):
   def __init__(self, **kwargs) -> None:
      input_embedding_dim    = kwargs.get('input_embedding_dim')
      num_latent_heads       = kwargs.get('point_latent_heads')
      self.num_latent_layers = kwargs.get('point_latent_layers')
      self.lr                = kwargs.get('lr_point')
      seed                   = kwargs.get('seed')
      set_seed(seed)
      super().__init__()

      # Input Feature Embedding
      self.heat_embedding = nn.Sequential(
         nn.Linear(1, input_embedding_dim//2),
         nn.ReLU(),          
         nn.Linear(input_embedding_dim//2, input_embedding_dim),
         nn.ReLU(),
      )

      # Latent trasnformer module
      self.self_latent_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_latent_heads, batch_first=True)
      self.self_latent_ff = FeedForward(input_embedding_dim)

      # Decoder
      self.output_layer = nn.Sequential(
               nn.Linear(input_embedding_dim, input_embedding_dim//2),
               nn.ReLU(),
               nn.Linear(input_embedding_dim//2, 1)
            )
   
   def forward(self, inputs, heat) :
      heat = heat.reshape(heat.shape[0], -1, 1)
      data = self.heat_embedding(heat)

      x = inputs + data

      for _ in range(self.num_latent_layers):
         x = self.self_latent_attention(x) 
         x = self.self_latent_ff(x) + x

      p = self.output_layer(x)
      return p

   def reset_train(self) :
      self.criterion = nn.CrossEntropyLoss()
      self.optimizer = Lamb(self.parameters(), lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

class HeatPerceiver(nn.Module):

   def __init__(self, **kwargs) -> None:
      '''
      Kwargs list :
      - no_points : number of points in the cloth (default: 400)
      - action_dim : number of action dimension _offset_ (default: 2)
      - hidden_dim : number of hidden dimension (default: 128)
      '''

      #super(TransformerModel, self).__init__()

      super().__init__()

      depth               = kwargs.get('depth', 3)
      input_dim           = kwargs.get('input_dim', 3)
      input_embedding_dim = kwargs.get('input_embedding_dim', 128)
      num_latents         = kwargs.get('num_latents', 50)
      num_cross_heads     = kwargs.get('num_cross_heads', 8)
      num_output_heads    = kwargs.get('num_output_heads', 8)
      num_latent_heads    = kwargs.get('num_latent_heads', 8)
      num_latent_layers   = kwargs.get('num_latent_layers', 3)
      lr_heat             = kwargs.get('lr_heat', 1e-4)

      #Point Network
      point_latent_heads   = kwargs.get('point_latent_heads', 8)
      point_latent_layers  = kwargs.get('point_latent_layers', 3)
      lr_point             = kwargs.get('lr_point', 1e-4)
      # Action parameters
      action_latent_heads  = kwargs.get('action_latent_heads', 8)
      action_cross_heads   = kwargs.get('action_cross_heads', 8)
      action_latent_layers = kwargs.get('action_latent_layers', 3)
      lr_action            = kwargs.get('lr_action', 1e-4)

      seed = kwargs.get('seed', None)
      set_seed(seed)

      self.use_gt_prediction = False

      

      self.init_params = dict(kwargs)
      

      self.heat_predictor = HeatPredictor(
                        depth = depth,
                        input_dim = input_dim,
                        input_embedding_dim = input_embedding_dim,
                        num_latents = num_latents,
                        num_cross_heads = num_cross_heads,
                        num_output_heads = num_output_heads,
                        num_latent_heads = num_latent_heads,
                        num_latent_layers = num_latent_layers,
                        lr_heat = lr_heat,
                        seed = seed
                        )

      self.point_prediction = PointPredictor(
                                 input_embedding_dim = input_embedding_dim,
                                 point_latent_heads = point_latent_heads,
                                 point_latent_layers = point_latent_layers,
                                 lr_point = lr_point,
                                 seed  = seed
                                 )

      self.action_predciton = ActionPredictor(
                        input_embedding_dim = input_embedding_dim,
                        action_cross_heads = action_cross_heads,
                        action_latent_heads = action_latent_heads,
                        action_latent_layers = action_latent_layers,
                        lr_action = lr_action
                        seed = seed
                        )


      self.best_val_loss = float('inf')


   def reset_train(self) :
      self.heat_predictor.reset_train()
      self.action_predciton.reset_train()

   def set_seed(self, seed):
      set_seed(seed)
      
   def forward(self, pts):


      b = pts.shape[0]
      
      heat = self.heat_predictor.forward(pts)

      pte = self.heat_predictor.input_embedding(pts)
            
      p = self.point_prediction.forward(pte, heat).squeeze(-1)

      id = torch.argmax(p, dim=-1)

      pt = pts[torch.arange(b),id,:2].unsqueeze(1)

      output = self.action_predciton.forward(pt, pte)

      return heat, p, output

   def forward_heat(self, inputs):
      inputs.to('cuda')
      h = self.heat_predictor(inputs)
      return h

   def forward_action(self, inputs):
      inputs.to('cuda')
      a = self.action_predciton(inputs)
      return a
   

   def trainer_heat(self, train_loader, device='cuda') :
      self.heat_predictor.reset_train()
      out = self.heat_predictor.trainer(train_loader, device=device)
      return out
   
   def validate_heat(self, val_loader, device='cuda') :
      out = self.heat_predictor.validate(val_loader, device=device)
      return out


   def trainer_pointprediction(self, train_loader, device = 'cuda') :

      self.train()
      total_train_loss = 0.0
      acc, precision_at_4, precision_at_3, precision_at_2, distance_prob = 0.0, 0.0, 0.0, 0.0, 0.0
      self.point_prediction.reset_train()


      for i, param in enumerate(self.heat_predictor.parameters()):
         if i == 0:
            param.requires_grad = False
            

      for i, data in enumerate(train_loader) :
         # get the inputs and labels from the data loader
         obs, gt = data

         pts = obs[0].to(device)
         #heat = obs[1].to(device)
         gt_prob = torch.zeros_like(gt[0]).to(device)
         gt_prob[torch.where(gt[0] == 1)] = 1

         b = pts.shape[0]
         npoins = pts.shape[1]

         # forward + backward + optimize
         heat = self.heat_predictor(pts)
         inputs = self.heat_predictor.input_embedding(pts)
         p = self.point_prediction(inputs, heat)

         p = p.reshape(b, -1)
         # Get loss function
         loss = self.point_prediction.criterion(p, gt_prob)

         acc += (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
         precision_at_4 += get_precision_at_k(5, p, gt_prob)
         precision_at_3 += get_precision_at_k(3, p, gt_prob)
         precision_at_2 += get_precision_at_k(2, p, gt_prob)

         pt_taken, pt_gt = torch.argmax(p, dim=1), torch.argmax(gt_prob, dim=1)
         pt_pos, gt_pos =  pts[torch.arange(b), pt_taken], pts[torch.arange(b), pt_gt]
         distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b


         # zero the parameter gradients
         self.point_prediction.optimizer.zero_grad()
         loss.backward()
         self.point_prediction.optimizer.step()

         total_train_loss += loss.item()

         # Log learning rate to Tensorboard
         lr = self.point_prediction.optimizer.param_groups[0]["lr"]
                         
      self.training_step = {'point_train_loss': total_train_loss / len(train_loader),
                           'accuracy' : acc / len(train_loader),
                           'precision5': precision_at_4 / len(train_loader),
                           'precision3': precision_at_3 / len(train_loader),
                           'precision2': precision_at_2 / len(train_loader),
                           'distance_prob': distance_prob / len(train_loader),
                           'lr': lr
                           }

      return self.training_step
   
   def validate_pointprediction(self, val_loader, device='cuda') :
      # validate the model
      total_val_loss = 0.0
      acc, precision_at_4, precision_at_3,precision_at_2, distance_prob = 0.0, 0.0, 0.0, 0.0, 0.0
      best_val_loss = float('inf')
      self.eval()
      with torch.no_grad():

         val_step = {}
         for i , data in enumerate(val_loader) :

            obs, gt = data

            pts = obs[0]

            b = pts.shape[0]
            npoins = pts.shape[1]

            gt_prob = torch.zeros_like(gt[0]).to(device)
            gt_prob[torch.where(gt[0] == 1)] = 1
            

            #Move to device
            pts = pts.to(device)
            gt_prob = gt_prob.to(device)


            # forward + backward + optimize
            heat =self.heat_predictor(pts)
            inputs = self.heat_predictor.input_embedding(pts)
            p = self.point_prediction(inputs, heat)
            p = p.reshape(b, -1)

            # Get loss function
            prob_loss = self.point_prediction.criterion(p, gt_prob)
            
            total_val_loss += prob_loss.item()

            #Get accuracy
            acc += (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
            precision_at_4 += get_precision_at_k(5, p, gt_prob)
            precision_at_3 += get_precision_at_k(3, p, gt_prob)
            precision_at_2 += get_precision_at_k(2, p, gt_prob)

            pt_taken, pt_gt = torch.argmax(p, dim=1), torch.argmax(gt_prob, dim=1)
            pt_pos, gt_pos =  pts[torch.arange(b), pt_taken], pts[torch.arange(b), pt_gt]
            distance_prob += torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b


      self.point_prediction.scheduler.step(total_val_loss)
      val_step = {'point_val_loss': total_val_loss/ len(val_loader),
                  'accuracy' : acc/len(val_loader),
                  'precision5': precision_at_4 / len(val_loader),
                  'precision3': precision_at_3 / len(val_loader),
                  'precision2': precision_at_2 / len(val_loader),
                  'distance_prob': distance_prob / len(val_loader),
                  }
      
      # Save Best Model
      if total_val_loss/ len(val_loader) < best_val_loss:
         self.best_train_loss = val_step['val_loss']
         best_val_loss = total_val_loss/ len(val_loader)

         
      return val_step



   def trainer_action(self, train_loader, device='cuda') :

      self.train()

      for i, param in enumerate(self.heat_predictor.parameters()):
         if i == 0:
            param.requires_grad = False
      for i, param in enumerate(self.point_prediction.parameters()):
         if i == 0:
            param.requires_grad = False

      total_train_loss = 0.0
      distance_d, distance_dx, distance_dy, angle = 0.0, 0.0, 0.0, 0.0

      self.action_predciton.reset_train()

      for i, data in enumerate(train_loader) :
      #for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Progress'):
         # get the inputs and labels from the data loader
         obs, gt = data


         pts = obs[0].to(device)
         gt_gaussian = gt[0]
         gt_action = gt[1].to(device)

         b = pts.shape[0]
         npoins = pts.shape[1]


         inputs = self.heat_predictor.input_embedding(pts)
         
         pt_taken = pts[torch.arange(b), torch.where(gt_gaussian == 1)[1], :2].unsqueeze(1)

         a = self.action_predciton.forward(pt_taken, inputs)
         a = a.reshape(b, -1)

         # Get loss function
         loss = self.action_predciton.criterion(a, gt_action[:,2:])

         # zero the parameter gradients
         self.action_predciton.optimizer.zero_grad()
         loss.backward()
         self.action_predciton.optimizer.step()

         total_train_loss += loss.item()

         # Log learning rate to Tensorboard
         lr = self.action_predciton.optimizer.param_groups[0]["lr"]

         distance_d += torch.sqrt(torch.sum((a - gt_action[:,2:])**2, dim=1)).sum().item() / b
         distance_dx += torch.sqrt(torch.sum((a[:,-2] - gt_action[:,-2])**2, dim=0)).sum().item() / b
         distance_dy += torch.sqrt(torch.sum((a[:,-1] - gt_action[:,-1])**2, dim=0)).sum().item() / b
         angle += torch.sqrt(torch.sum((torch.atan2(a[:,-1], a[:,-2]) - torch.atan2(gt_action[:,-1], gt_action[:,-2]))**2, dim=0)).sum().item() / b

      self.training_step = {'action_train_loss': total_train_loss / len(train_loader),
                           'distance_d': distance_d / len(train_loader),
                           'distance_dx': distance_dx / len(train_loader),
                           'distance_dy': distance_dy / len(train_loader),
                           'angle': angle / len(train_loader),
                           'lr_action': lr
                           }
      
      return self.training_step

   def validate_action(self, val_loader, device='cuda') :
      total_val_loss = 0.0
      distance_d, distance_dx, distance_dy,angle = 0.0, 0.0, 0.0, 0.0

      best_val_loss = float('inf')

      self.eval()
      with torch.no_grad():

         val_step = {}
         for i, data in enumerate(val_loader) :   
            obs, gt = data

            pts = obs[0].to(device)
            gt_gaussian = gt[0].to(device)
            gt_action = gt[1].to(device)

            b = pts.shape[0]
            
            inputs = self.heat_predictor.input_embedding(pts)
            pt_taken = pts[torch.arange(b), torch.where(gt_gaussian == 1)[1], :2].unsqueeze(1)

            a = self.action_predciton.forward(pt_taken, inputs)
            a = a.reshape(b, -1)
            # Get loss function
            loss = self.action_predciton.criterion(a, gt_action[:,2:])

            # zero the parameter gradients
            total_val_loss += loss.item()


            distance_d += torch.sqrt(torch.sum((a - gt_action[:,2:])**2, dim=1)).sum().item() / b
            distance_dx += torch.sqrt(torch.sum((a[:,-2] - gt_action[:,-2])**2, dim=0)).sum().item() / b
            distance_dy += torch.sqrt(torch.sum((a[:,-1] - gt_action[:,-1])**2, dim=0)).sum().item() / b
            angle += torch.sqrt(torch.sum((torch.atan2(a[:,-1], a[:,-2]) - torch.atan2(gt_action[:,-1], gt_action[:,-2]))**2, dim=0)).sum().item() / b


         self.action_predciton.scheduler.step(total_val_loss)
         val_step = {'val_loss_action': total_val_loss / len(val_loader),
                              'distance_d': distance_d / len(val_loader),
                              'distance_dx': distance_dx / len(val_loader),
                              'distance_dy': distance_dy / len(val_loader),
                              'angle': angle / len(val_loader),
                              }
         
         # Save Best Model
         if total_val_loss/ len(val_loader) < best_val_loss:
            self.best_train_loss = val_step['val_loss_action']
            best_val_loss = total_val_loss/ len(val_loader)

         
      return val_step
   

   def test(self, test_loader, device='cuda') :
      distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0
      distance_d, distance_dx, distance_dy = 0.0, 0.0, 0.0

      test_results = []   
      self.eval()
      with torch.no_grad():
         for i, data in enumerate(test_loader) :   
            obs, gt = data


            b = obs[0].shape[0]
            pts = obs[0].to(device)
            gt_action = gt[1].to(device)

            h,a = self.forward(pts)
            a = a.reshape(b, -1)

            distance    = torch.sqrt(torch.sum((a[:,:2] - gt_action[:,:2])**2, dim=1)).sum().item() / b
            distance_x  = torch.sqrt(torch.sum((a[:,0] - gt_action[:,0])**2, dim=0)).sum().item() / b
            distance_y  = torch.sqrt(torch.sum((a[:,1] - gt_action[:,1])**2, dim=0)).sum().item() / b


            distance_d  = torch.sqrt(torch.sum((a[:,2:] - gt_action[:,2:])**2, dim=1)).sum().item() / b
            distance_dx = torch.sqrt(torch.sum((a[:,-2] - gt_action[:,-2])**2, dim=0)).sum().item() / b
            distance_dy = torch.sqrt(torch.sum((a[:,-1] - gt_action[:,-1])**2, dim=0)).sum().item() / b
            angle       = torch.sqrt(torch.sum((torch.atan2(a[:,-1], a[:,-2]) - torch.atan2(gt_action[:,-1], gt_action[:,-2]))**2, dim=0)).sum().item() / b


            result = {  'distance': distance / len(test_loader),
                        'distance_x': distance_x / len(test_loader),
                        'distance_y': distance_y / len(test_loader),
                        'distance_d': distance_d / len(test_loader),
                        'distance_dx': distance_dx / len(test_loader),
                        'distance_dy': distance_dy / len(test_loader),
                        'angle': angle / len(test_loader),
                        }
            test_results.append(result)

      return test_results
   
 

   def save_best_model(self, path):

      state = {'init_params': self.init_params,
               'heat_model': self.heat_predictor.state_dict(),
               'point_model': self.point_prediction.state_dict(),
               'action_model': self.action_predciton.state_dict(),
      }

      #get folder path minus the file name
      folder_path = os.path.dirname(path)
      # create folder if it doesn't exist
      print(folder_path)
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)

      torch.save(state, path)

   def get_best_result(self):
      return {'best_train_loss': self.best_train_loss, 'best_val_loss': self.best_val_loss}



   def get_matplotlib_heatmap(self, heat, gaussian_tensor, permuted_indices):

      plt.clf()
      fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))


      original_order_indices = torch.argsort(permuted_indices)
      heat_ordered = heat[:, original_order_indices,:]
      heat_ordered = heat_ordered.reshape(25,25).detach().cpu().numpy().transpose(1,0)
      heat = heat.reshape(25,25).detach().cpu().numpy().transpose(1,0)

      gaussian_tensor = gaussian_tensor.detach().cpu().numpy().reshape(25,25).transpose(1,0)
      gaussian_permuted  = gaussian_tensor.reshape(-1)[permuted_indices].reshape(25,25).transpose(1,0)

      # Define a custom colormap that goes from white to red
      cmap = mcolors.LinearSegmentedColormap.from_list("white_to_red", [(1, 1, 1), (1, 0, 0)], N=256)

      # Plot the first image (gt) on the left
      axs[0,0].imshow(gaussian_tensor, origin='lower', cmap=cmap, vmin=0, vmax=1)
      axs[0,0].set_title("Ground Truth (GT) ordered")
      axs[0,1].imshow(gaussian_permuted, origin='lower', cmap=cmap, vmin=0, vmax=1)
      axs[0,1].set_title("Ground Truth")

      axs[1,0].imshow(heat_ordered, origin='lower', cmap=cmap, vmin=0, vmax=1)
      axs[1,0].set_title("Prediction ordered")
      axs[1,1].imshow(heat, origin='lower', cmap=cmap, vmin=0, vmax=1)
      axs[1,1].set_title("Prediction")

      plt.tight_layout()

      return fig



   def get_matplotlib_heatmap2(self,ptc, heat, gaussian_tensor):

      plt.clf()
      fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))

      heat = heat.detach().cpu().numpy()
      ptc = ptc.detach().cpu().numpy()
      gaussian_tensor = gaussian_tensor.detach().cpu().numpy().reshape(25,25).transpose(1,0)

      # Define a custom colormap that goes from white to red
      cmap = mcolors.LinearSegmentedColormap.from_list("white_to_red", [(1, 1, 1), (1, 0, 0)], N=256)

      # Plot the first image (gt) on the left
      axs[0].imshow(gaussian_tensor, origin='lower', cmap=cmap, vmin=0, vmax=1)
      axs[0].set_title("Ground Truth (GT) ordered")

      axs[1].scatter(ptc[0,:,0], ptc[0,:,1], c=heat)
      axs[1].set_title("Prediction ordered")

      plt.tight_layout()

      return fig


   def get_matplotlib_point(self, pts, pt_taken, pt_gt) :

      pts_numpy = pts.detach().cpu().squeeze(0).numpy()
      plt.clf()
      fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))

      axs[0].scatter(pts_numpy[:,0], pts_numpy[:,1], c='b')
      axs[0].scatter(pts_numpy[pt_gt,0], pts_numpy[pt_gt,1], c='r', s=200)
      #axs[1,0].plot((pt_taken[0,0], pt_taken[0,0] + a_gt[0,-2]),(pt_taken[0,1], pt_taken[0,1] + a_gt[0,-1]), c='g')
      axs[0].set_title("Diff GT")

      #axs[1,0].arrow(x=pt_taken[0,0],y= pt_taken[0,1], dx=a_gt[0,-2], dy=a_gt[0,-1], width=0.004)

      axs[1].scatter(pts_numpy[:,0], pts_numpy[:,1], c='b')
      axs[1].scatter(pts_numpy[pt_taken,0], pts_numpy[pt_taken,1], c='r', s=200)
      #axs[1,1].plot((a[0,0], a[0,0] + a[0,-2]),(a[0,1], a[0,1] + a[0,-1]), c='g')
      axs[1].set_title("Diff")
      plt.tight_layout()
      return fig



   def get_matplotlib_action(self, pts, pt_id, gt_id, a, a_gt) :

      pts_numpy = pts.reshape(-1,3).detach().cpu().squeeze(0).numpy()
      a = a.reshape(2).detach().cpu().squeeze(0).numpy()
      a_gt = a_gt.reshape(2).detach().cpu().squeeze(0).numpy()

      plt.clf()
      fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))

      pt_taken = np.reshape(pts_numpy[pt_id, :], (3))
      pt_gt    = np.reshape(pts_numpy[gt_id, :], (3))

      axs[0].scatter(pts_numpy[:,0], pts_numpy[:,1], c='b')
      axs[0].scatter(pt_gt[0], pt_gt[1], c='r', s=200)
      #axs[0].plot((pt_gt[0], pt_gt[0] + a_gt[-2]),(pt_gt[1], pt_gt[1] + a_gt[-1]), c='g')
      axs[0].arrow(x=pt_gt[0],y= pt_gt[1], dx=a_gt[-2], dy=a_gt[-1], width=0.004)
      axs[0].set_title("Diff GT")


      axs[1].scatter(pts_numpy[:,0], pts_numpy[:,1], c='b')
      axs[1].scatter(pt_taken[0], pt_taken[1], c='r', s=200)
      #axs[0].plot((pt_taken[0], pt_taken[0] + a[-2]),(pt_taken[1], pt_taken[1] + a[-1]), c='g')
      axs[1].arrow(x=pt_taken[0],y= pt_taken[1], dx=a[-2], dy=a[-1], width=0.004)
      axs[1].set_title("Diff")
      plt.tight_layout()
      return fig


   def get_matplotlib_action2(self, pts, pts_pointcloud, pt_id, gt_id, a, a_gt) :

      pts_numpy = pts.reshape(-1,3).detach().cpu().squeeze(0).numpy()
      pts_numpy2 = pts_pointcloud.reshape(-1,3).detach().cpu().squeeze(0).numpy()
      a = a.reshape(2).detach().cpu().squeeze(0).numpy()
      a_gt = a_gt.reshape(2).detach().cpu().squeeze(0).numpy()

      plt.clf()
      fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))

      pt_taken = np.reshape(pts_numpy2[pt_id, :], (3))
      pt_gt    = np.reshape(pts_numpy[gt_id, :], (3))

      axs[0].scatter(pts_numpy[:,0], pts_numpy[:,1], c='b')
      axs[0].scatter(pt_gt[0], pt_gt[1], c='r', s=200)
      #axs[0].plot((pt_gt[0], pt_gt[0] + a_gt[-2]),(pt_gt[1], pt_gt[1] + a_gt[-1]), c='g')
      axs[0].arrow(x=pt_gt[0],y= pt_gt[1], dx=a_gt[-2], dy=a_gt[-1], width=0.004)
      axs[0].set_title("Diff GT")


      axs[1].scatter(pts_numpy2[:,0], pts_numpy2[:,1], c='b')
      axs[1].scatter(pt_taken[0], pt_taken[1], c='r', s=200)
      #axs[0].plot((pt_taken[0], pt_taken[0] + a[-2]),(pt_taken[1], pt_taken[1] + a[-1]), c='g')
      axs[1].arrow(x=pt_taken[0],y= pt_taken[1], dx=a[-2], dy=a[-1], width=0.004)
      axs[1].set_title("Diff")
      plt.tight_layout()
      return fig

   def get_heatmap_image(self, heat, gaussian_tensor, gaussian_permuted):
      
      heat = heat.reshape(25,25).detach().cpu().numpy()
      gaussian_tensor = gaussian_tensor.detach().cpu().numpy()
      gaussian_permuted  = gaussian_permuted.reshape(25,25).detach().cpu().numpy()

      heat = (heat * 255).astype(np.uint8)
      gaussian_tensor = (gaussian_tensor * 255).astype(np.uint8)
      gaussian_permuted = (gaussian_permuted * 255).astype(np.uint8)

      blank_column = np.ones((25, 10), dtype=np.uint8) * 255
      concatenated_heatmap = np.concatenate((gaussian_tensor, blank_column, gaussian_permuted, blank_column, heat), axis=1)
      concatenated_image = Image.fromarray(concatenated_heatmap, mode='L')

      return concatenated_image
   
      return image
      fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))

      # Define a custom colormap that goes from white to red
      cmap = mcolors.LinearSegmentedColormap.from_list("white_to_red", [(1, 1, 1), (1, 0, 0)], N=256)

      # Plot the first image (gt) on the left
      axs[0].imshow(heatmap_gt, origin='lower', cmap=cmap, vmin=0, vmax=1)
      axs[0].set_title("Ground Truth (GT) ordered")
      axs[1].imshow(heatmap_gt[:, indices, 0], origin='lower', cmap=cmap, vmin=0, vmax=1)
      axs[1].set_title("Ground Truth Input")
      axs[2].imshow(heatmap_p, origin='lower', cmap=cmap, vmin=0, vmax=1)
      axs[2].set_title("Ground Truth Input")

      return fig
      # Concatenate the two heatmaps with a blank column in between
      blank_column = np.ones((heatmap_gt.shape[0], 1, 1), dtype=np.uint8) * 0
      concatenated_heatmap = np.concatenate((heatmap_gt, blank_column, heatmap_gt, blank_column, heatmap_p), axis=1)

      print(concatenated_heatmap.shape)

      concatenated_image = Image.fromarray(np.squeeze(concatenated_heatmap, -1), mode='L')

      return concatenated_image

class ActionPredictor(nn.Module):

   def __init__(self, **kwargs) -> None:
      super().__init__()

      input_embedding_dim    = kwargs.get('input_embedding_dim')
      num_cross_heads        = kwargs.get('action_cross_heads')
      num_latent_heads       = kwargs.get('action_latent_heads')
      self.num_latent_layers = kwargs.get('action_latent_layers')
      self.lr                = kwargs.get('lr_action')
      seed = kwargs.get('seed', None)
      set_seed(seed)
      
      # Input Feature Embedding
      self.pts_embedding = nn.Sequential(
         nn.Linear(2, input_embedding_dim//2),
         nn.ReLU(),          
         nn.Linear(input_embedding_dim//2, input_embedding_dim),
         nn.ReLU(),
      )

      # Input cross attention between points (Q) and point taken (K,V)
      self.cross_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_cross_heads, batch_first=True)
      self.cross_ff = FeedForward(input_embedding_dim)

      # Latent trasnformer module
      self.self_latent_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_latent_heads, batch_first=True)
      self.self_latent_ff = FeedForward(input_embedding_dim)

      self.output_layer = nn.Sequential(
         nn.Linear(input_embedding_dim, input_embedding_dim//2),
         nn.ReLU(),
         nn.Linear(input_embedding_dim//2, 2)
      )


   def forward(self, pt_taken, pts) :
      
      pts_e = self.pts_embedding(pt_taken)

      x = self.cross_attention(pts_e, context = pts)
      x = self.cross_ff(x) + x

      for _ in range(self.num_latent_layers):
         x = self.self_latent_attention(x) 
         x = self.self_latent_ff(x) + x

      a = self.output_layer(x)

      return a.tanh()
   


   def reset_train(self) :
      self.criterion = nn.MSELoss()
      self.optimizer = Lamb(self.parameters(), lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')






if __name__ == '__main__' :

   hparams = {'num_epochs' : 150,
              'batch_size' : 128,            
               'val_ratio'  : 0.21,
               'test_ratio' : 0.15,
               'seed' : 42,

               #Heat Predictor Perceiver
               'depth' : 3,
               'input_dim' : 3,
               'input_embedding_dim' : 128,
               'num_latents' : 50,
               'num_cross_heads' : 8,
               'num_output_heads' : 8,
               'num_latent_heads' : 8,
               'num_latent_layers' : 3,
               'lr_heat' : 1e-4,

               #Point Network
               'point_latent_heads' : 8,
               'point_latent_layers' : 3,
               'lr_point' : 1e-4,

               # Action parameters
               'action_latent_heads' : 8,
               'action_cross_heads' : 8,
               'action_latent_layers' : 3,
               'lr_action' : 1e-4,
               }


   agent = HeatPerceiver(**hparams)



   b = 10
   pts = torch.rand(b,120, 3)
   
   heat = agent.heat_predictor.forward(pts)
   print('heat', heat.shape)

   pte = agent.heat_predictor.input_embedding(pts)
   print('inputs', pte.shape)
         
   p = agent.point_prediction.forward(pte, heat).squeeze(-1)
   print('p', p.shape)

   id = torch.argmax(p, dim=-1)
   print('id', id.shape)

   pt = pts[torch.arange(b),id,:2].unsqueeze(1)
   print('pt', pt.shape)

   output = agent.action_predciton.forward(pt, pte)

   print('output', output.shape)
   
   print('Simulate')
   train_loader = []
   for _ in range(10):
      p  = torch.randint(0, 500, (1,)).item()
      obs = (torch.rand(10,500,3), torch.rand(10,500))
      gt = (torch.rand(10,500), torch.rand(10,4))

      obs[1][:,p] = 1
      gt[0][:,p] = 1

      train_loader.append((obs, gt))

   agent.to('cuda')
   agent.trainer_heat(train_loader)
   agent.validate_heat(train_loader)
   print('Heat ok')
   agent.trainer_pointprediction(train_loader)
   agent.validate_pointprediction(train_loader)
   print('Point ok')
   agent.trainer_action(train_loader)
   agent.validate_action(train_loader)
   print('Action ok')