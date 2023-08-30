import torch
from cloth_training.dataset.dataset_gen import GymClothDataset
from cloth_training.model.paper.dual_perceiver import DualPerceiver

import os, pickle, time
from tqdm import tqdm
from torch.utils.data import DataLoader
from cloth_training.model.common.model_utils import set_seed
import copy 
import matplotlib.pyplot as plt  

if __name__ == '__main__' :

   folder_name = 'base_transformer'
   write_log = True
   save_model = True
   save_model_path = './saved_model'
   dataset_path = './dataset/ablation/ablation.pt'

   #load param from file

   param = './saved_model/dual_perceiver-08-26-15-36_hparams.pickle'
   model = './saved_model/dual_perceiver-08-26-15-36.pth'

   with open(param, 'rb') as f :
      param = pickle.load(f)

   set_seed(param['seed'])
   # Iterating over all combinations of hyperparameters
   dataset = torch.load(dataset_path)
   dataset.set_obs_type('heatmap')
   dataset.set_output_type('heatmap')
   dataset.to_device(torch.device('cpu'))
   #dataset.shuffle_points()

   val_sample  = int(len(dataset) * param['val_ratio'])
   train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_sample, val_sample])
   
   train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)
   val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=True)


   # MODEL CREATION
   agent = DualPerceiver(**param)
   
   model = torch.load(model)
   agent.point_prediction.load_state_dict(model['point_model'])
   agent.action_predciton.load_state_dict(model['action_model'])
   agent.to('cuda:0')
   print('Load model ok')
   ### LOG ##
   run_id = folder_name + '-'  + str(time.strftime("%m-%d-%H-%M"))

   ###### 1) TEST TRANSFORMER #####
   device = 'cuda'
   agent.eval()
   save_path = './test/transformer_dual_perceiver'
   os.makedirs(save_path, exist_ok=True)
   with torch.no_grad():

      for i , data in enumerate(val_loader) :
         print('Sample : ', i)
         obs, gt = data

         b = obs[0].shape[0]

         pts = obs[0]

         pippo = pts.reshape(1, -1, 3).to(device)
         p,a = agent.forward(pippo)

         pts = pts.reshape(-1,3).cpu()
         p = p.reshape(-1).cpu()
         a = a.reshape(2).cpu()

         id_taken = torch.argmax(p, dim=0)
         pt = pts[id_taken,:2].reshape(2).numpy()  
         action = a.numpy()
         pts = pts.cpu().reshape(-1,3).numpy()
         
         plt.scatter(pts[:,0], pts[:,1], c='b')
         plt.scatter(pt[0], pt[1], c='r', s=200)
         plt.arrow(x=pt[0],y= pt[1], 
                   dx=action[-2], dy=action[-1], width=0.010, color='r')
         plt.tight_layout()
         file_name = f'validation_ep_{i}_action.png'
         plt.savefig(os.path.join(save_path, file_name))
         plt.close()

