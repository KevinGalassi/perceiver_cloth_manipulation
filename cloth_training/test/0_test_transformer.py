import torch
from cloth_training.dataset.dataset_gen import GymClothDataset
from cloth_training.model.paper.transformer import DaggerTransformer

import os, pickle, time
from tqdm import tqdm
from torch.utils.data import DataLoader
from cloth_training.model.common.model_utils import set_seed

import matplotlib.pyplot as plt  

if __name__ == '__main__' :

   folder_name = 'base_transformer'
   write_log = True
   save_model = True
   save_model_path = './saved_model'
   dataset_path = './dataset/ablation/ablation.pt'

   #load param from file

   param = './saved_model/base_transformer/base_transformer-08-30-15-22_hparams.pickle'
   model = './saved_model/base_transformer/base_transformer-08-30-15-22.pth'

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
   agent = DaggerTransformer(**param)
   agent.load_state_dict(torch.load(model)['model'])
   agent.to('cuda:0')
   print('Load model ok')
   ### LOG ##
   run_id = folder_name + '-'  + str(time.strftime("%m-%d-%H-%M"))

   ###### 1) TEST TRANSFORMER #####
   device = 'cuda'
   agent.eval()
   save_path = './test/transformer'
   os.makedirs(save_path, exist_ok=True)
   with torch.no_grad():

      val_step = {}
      for i , data in enumerate(val_loader) :
         print('Sample : ', i)
         obs, gt = data

         b = obs[0].shape[0]
         pts = obs[0].to(device)
         gt_gaussian = gt[0].to(device)
         gt_action = gt[1].to(device)

         output = agent.forward(pts)

         pts = pts.reshape(-1,3).cpu().numpy()
         pt = (output.reshape(4)[:2].cpu().numpy()/2) + 0.5
         a  = output.reshape(4)[2:].cpu().numpy()
         plt.scatter(pts[:,0], pts[:,1], c='b')
         plt.scatter(pt[0], pt[1], c='r', s=200)
         plt.arrow(x=pt[0],y= pt[1], 
                   dx=a[-2], dy=a[-1], width=0.010, color='r')
         plt.tight_layout()
         file_name = f'validation_ep_{i}_action.png'
         plt.savefig(os.path.join(save_path, file_name))
         plt.close()

