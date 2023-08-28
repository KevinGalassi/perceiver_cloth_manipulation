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


class PointPredictor(nn.Module):
   def __init__(self, **kwargs) -> None:
      input_embedding_dim  = kwargs.get('input_embedding_dim')
      num_latent_heads     = kwargs.get('point_latent_heads')
      self.num_layers      = kwargs.get('point_layers')
      self.lr              = kwargs.get('lr_point')
      seed                 = kwargs.get('seed')
      set_seed(seed)
      super().__init__()


      # Input Feature Embedding
      self.input_embedding = nn.Sequential(
         nn.Linear(3, input_embedding_dim//2),
         nn.ReLU(),          
         nn.Linear(input_embedding_dim//2, input_embedding_dim),
         nn.ReLU(),
      )


      # Latent trasnformer module
      #self.self_latent_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_latent_heads, batch_first=True)
      #self.self_latent_ff = FeedForward(input_embedding_dim)
      self.transformer_layers = nn.ModuleList()
      for _ in range(self.num_layers):
         self.transformer_layers.append(Attention(embed_dim=input_embedding_dim, num_heads=num_latent_heads, batch_first=True))
         self.transformer_layers.append(FeedForward(input_embedding_dim))


      # Decoder
      self.output_layer = nn.Sequential(
               nn.Linear(input_embedding_dim, input_embedding_dim//2),
               nn.ReLU(),
               nn.Linear(input_embedding_dim//2, 1)
            )
   
   def forward(self, pts) :
      x = self.input_embedding(pts)
      for i in range(0, len(self.transformer_layers), 2):
         x = self.transformer_layers[i](x)
         x = self.transformer_layers[i+1](x) + x

      p = self.output_layer(x)
      return p

   def reset_train(self) :
      self.criterion = nn.CrossEntropyLoss()
      self.optimizer = Lamb(self.parameters(), lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

class ActionPredictor(nn.Module):

   def __init__(self, **kwargs) -> None:
      super().__init__()

      input_embedding_dim    = kwargs.get('input_embedding_dim')
      num_cross_heads        = kwargs.get('action_cross_heads')
      num_latent_heads       = kwargs.get('action_latent_heads')
      self.num_latent_layers = kwargs.get('action_latent_layers')
      self.depth             = kwargs.get('action_depth')
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
      #self.self_latent_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_latent_heads, batch_first=True)
      #self.self_latent_ff = FeedForward(input_embedding_dim)

      self.transformer_layers = nn.ModuleList()
      for _ in range(self.num_latent_layers):
         self.transformer_layers.append(Attention(embed_dim=input_embedding_dim, num_heads=num_latent_heads, batch_first=True))
         self.transformer_layers.append(FeedForward(input_embedding_dim))

      self.output_layer = nn.Sequential(
         nn.Linear(input_embedding_dim, input_embedding_dim//2),
         nn.ReLU(),
         nn.Linear(input_embedding_dim//2, 2)
      )


   def forward(self, pt_taken, pts) :
      
      x = self.pts_embedding(pt_taken)

      for _ in range(self.depth):
         x = self.cross_attention(x, context = pts)
         x = self.cross_ff(x) + x
         for i in range(0, len(self.transformer_layers), 2):
            x = self.transformer_layers[i](x)
            x = self.transformer_layers[i+1](x) + x

      a = self.output_layer(x)

      return a.tanh()
   


   def reset_train(self) :
      self.criterion = nn.MSELoss()
      self.optimizer = Lamb(self.parameters(), lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')



class DualPerceiver(nn.Module):

   def __init__(self, **kwargs) -> None:
      '''
      Kwargs list :
      - no_points : number of points in the cloth (default: 400)
      - action_dim : number of action dimension _offset_ (default: 2)
      - hidden_dim : number of hidden dimension (default: 128)
      '''

      #super(TransformerModel, self).__init__()

      super().__init__()

      input_dim           = kwargs.get('input_dim')
      input_embedding_dim = kwargs.get('input_embedding_dim')


      #Point Network
      point_latent_heads   = kwargs.get('point_latent_heads')
      point_layers         = kwargs.get('point_layers')
      lr_point             = kwargs.get('lr_point')
      # Action parameters
      action_depth         = kwargs.get('action_depth')
      action_latent_heads  = kwargs.get('action_latent_heads')
      action_cross_heads   = kwargs.get('action_cross_heads')
      action_latent_layers = kwargs.get('action_latent_layers')
      lr_action            = kwargs.get('lr_action')
      seed = kwargs.get('seed', None)
      set_seed(seed)

      self.use_gt_prediction = False
      self.init_params = dict(kwargs)
      
      self.point_prediction = PointPredictor(
                                 input_dim = input_dim,
                                 input_embedding_dim = input_embedding_dim,
                                 point_latent_heads = point_latent_heads,
                                 point_layers = point_layers,
                                 lr_point = lr_point,
                                 seed  = seed
                                 )

      self.action_predciton = ActionPredictor(
                        input_embedding_dim = input_embedding_dim,
                        action_cross_heads = action_cross_heads,
                        action_latent_heads = action_latent_heads,
                        action_latent_layers = action_latent_layers,
                        action_depth = action_depth,
                        lr_action = lr_action,
                        seed = seed
                        )


      self.best_val_loss = float('inf')


   def reset_train(self) :
      self.point_prediction.reset_train()
      self.action_predciton.reset_train()

   def set_seed(self, seed):
      set_seed(seed)
      
   def forward(self, pts):

      b = pts.shape[0]
            
      p = self.point_prediction.forward(pts).squeeze(-1)

      id = torch.argmax(p, dim=-1)
      pt = pts[torch.arange(b),id,:2].unsqueeze(1)
      pte = self.point_prediction.input_embedding(pts)
      
      output = self.action_predciton.forward(pt, pte)

      return p, output




   def trainer_pointprediction(self, train_loader, device = 'cuda') :

      self.train()
      total_train_loss = 0.0
      acc, precision_at_4, precision_at_3, precision_at_2, distance_prob = 0.0, 0.0, 0.0, 0.0, 0.0
      self.point_prediction.reset_train()


      for i, data in enumerate(train_loader) :
         # get the inputs and labels from the data loader
         obs, gt = data

         pts = obs[0].to(device)
         gt_prob = torch.zeros_like(gt[0]).to(device)
         gt_prob[torch.where(gt[0] == 1)] = 1

         b = pts.shape[0]
         npoins = pts.shape[1]

         # forward + backward + optimize
         p = self.point_prediction(pts)

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
            p = self.point_prediction(pts)
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
         self.best_train_loss = val_step['point_val_loss']
         best_val_loss = total_val_loss/ len(val_loader)

         
      return val_step



   def trainer_action(self, train_loader, device='cuda') :

      self.train()


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


         inputs = self.point_prediction.input_embedding(pts)
         
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
            
            inputs = self.point_prediction.input_embedding(pts)
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




if __name__ == '__main__' :

   hparams = {'num_epochs' : 150,
              'batch_size' : 128,            
               'val_ratio'  : 0.21,
               'test_ratio' : 0.15,
               'seed' : 42,

               'input_dim' : 3,
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


   agent = DualPerceiver(**hparams)



   b = 10
   pts = torch.rand(b,120, 3)
            
   p = agent.point_prediction.forward(pts).squeeze(-1)
   print('p', p.shape)

   id = torch.argmax(p, dim=-1)
   print('id', id.shape)

   pt = pts[torch.arange(b),id,:2].unsqueeze(1)
   print('pt', pt.shape)

   pte = agent.point_prediction.input_embedding(pts)
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
   agent.trainer_pointprediction(train_loader)
   agent.validate_pointprediction(train_loader)
   print('Point ok')
   agent.trainer_action(train_loader)
   agent.validate_action(train_loader)
   print('Action ok')