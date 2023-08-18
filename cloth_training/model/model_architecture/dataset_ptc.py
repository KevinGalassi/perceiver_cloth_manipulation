import torch
from torch.utils.data import Dataset
import torch.nn as nn

from tqdm import tqdm

import numpy as np
import os, pickle



from cloth_training.utils.pointcloud_utils import *
from cloth_training.utils.feature_extraction_utils import compute_surface_curvature

from cloth_training.model.model_architecture.model_utils import set_seed


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OBS_TYPE = ['rgb', 'd', 'rgbd', 'pts', 'heatmap', 'all', 'ptc', 'heatmap+ptc', 'vit']
OUTPUT_TYPE = ['prob', 'prob_gaussian', 'offset', 'heatmap','ptc','heatmap+ptc', 'vit']


PKL_LOAD_PATH      = '/home/kvn/code/cloth/cloth_training/dataset/prova.pkl'
PT_LIST_SAVE_PATH  = '/home/kvn/code/cloth/cloth_training/dataset/ptc/pt_list.pt'
DATASET_SAVE_PATH  = '/home/kvn/code/cloth/cloth_training/dataset/ptc/dataset.pt'
DATASET_SAVE_PATH2 = '/home/kvn/code/cloth/cloth_training/dataset/ptc/dataset2.pt'
GENERATE_LIST = True
CREATE_DATASET = True
EXTEND_DATASET = False

class GymClothDataset(Dataset):
   def __init__(self, action_list, obs_list, pts_list,
                obs_type = 'rgb', # 'rgb','d', pts', 'all'
                output_type = 'prob', #'prob, 'prob_gaussian', 'offset'
                pts_random = False,
                seed = 42,
                ptc_rdn_point = 10000,
                ptc_fps_point = 3000):
      

      action_list = action_list[:5]
      obs_list = obs_list[:5]
      pts_list = pts_list[:5]
      #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.pts_random = pts_random
      set_seed(seed) if seed is not None else None
      
      if obs_type not in OBS_TYPE : raise ValueError(f'obs_type must be in {OBS_TYPE}')
      self.obs_type = obs_type         
      if output_type not in OUTPUT_TYPE : raise ValueError(f'output_type must be in {OUTPUT_TYPE}')
      self.output_type = output_type

      self.ptc_rdn_point, self.ptc_fps_point = ptc_rdn_point, ptc_fps_point

      #stack list of tensor
      self.actions = torch.stack(action_list)
      self.obs     = torch.stack(obs_list)
      self.pts     = torch.stack(pts_list)
      
      self.n_sample = self.actions.shape[0]
      self.n_pts    = self.pts.shape[1]


      ################################################################
      # Find the match between grasp point and cloth point
      self.action_prob = self.action_prob_matching(self.actions, self.pts)

      #################################################################
      # Generate point cloud and features
      self.ptcs, self.background_idxs, self.cloth_idxs, self.features = self.get_pointcloud(self.pts, self.obs)
      
      # Get Heatmap for mesh pts and poitnclouds
      self.actions_gaussian, self.heatmap_ptc = self.get_gaussian_probability(self.action_prob, self.pts, self.ptcs, self.background_idxs, self.cloth_idxs) 
   
      print('Dataset length : ', self.n_sample)
      print('Action prob. shape : ', self.action_prob.shape)
      print('Action Gaussian', self.actions_gaussian.shape)
      print('Action shape : ', self.actions.shape)
      print('Obs. shape : ', self.obs.shape)
      print('Pts. shape : ', self.pts.shape)
      print('Ptc. shape : ', self.ptcs.shape)
      print('Features shape : ', self.features.shape)
      print('+++++++++++++')

   
   
   def __len__(self):
      return self.obs.shape[0]
   
   def __getitem__(self, index):
      obs = None
      out = None

      raise NotImplementedError('Not implemented yet')
      return obs, out


   def extend(self, action_list, obs_list, pts_list, ptc_fps_point=3000, ptc_rdn_point=10000) :
      actions = torch.stack(action_list)
      obs     = torch.stack(obs_list)
      pts     = torch.stack(pts_list)
      
      self.n_sample = actions.shape[0]

      action_prob = self.action_prob_matching(actions, pts)
      ptcs, background_idxs, cloth_idxs, features = self.get_pointcloud(pts, obs)
      actions_gaussian, heatmap_ptc = self.get_gaussian_probability(action_prob, pts, ptcs, background_idxs, cloth_idxs) 


      self.actions          = torch.cat((self.actions, actions), dim=0)
      self.actions_gaussian = torch.cat((self.actions_gaussian, actions_gaussian), dim=0)
      self.obs              = torch.cat((self.obs, obs), dim=0)
      self.pts              = torch.cat((self.pts, pts), dim=0)
      self.action_prob      = torch.cat((self.action_prob, action_prob), dim=0)
      self.ptc              = torch.cat((self.ptc, ptc), dim=0)
      self.features         = torch.cat((self.features, features), dim=0)
      self.heatmap_ptc      = torch.cat((self.heatmap_ptc, heatmap_ptc), dim=0)
      


   def action_prob_matching(self, actions, pts) :
      '''
      Given PTS point find where the action is been applied from the list
      '''
      act = (actions[:,:2] / 2.0) + 0.5
      distances2 = torch.norm(pts[:,:,:2]  - act[:,:2].unsqueeze(1), dim=2)
      min_distance_idx2 = torch.argmin(distances2, dim=1)
      action_prob = torch.zeros((self.n_sample, self.n_pts))
      action_prob[torch.arange(self.n_sample), min_distance_idx2] = 1
      return action_prob


   def get_pointcloud(self, pts, obs, ptc_rdn_point= 10000, ptc_fps_point= 3000) :
      ptcs = torch.empty(self.n_sample, ptc_fps_point, 3)
      features = torch.empty(self.n_sample, ptc_fps_point, 6)
      background_idxs = []
      cloth_idxs = []

      min = 1.4 - torch.max(pts[:, :, 2], dim=1)[0].unsqueeze(1).float()
      max = 1.4 - torch.min(pts[:, :, 2], dim=1)[0].unsqueeze(1).float()

      for i in tqdm(range(self.n_sample), total=self.n_sample):
         rgb = obs[i,:,:,:3]
         d   = obs[i,:,:,3]
         ptc = depth_to_point_cloud(d, max=max[i], min=min[i])
         
         rdn = rdn_index(ptc, ptc_rdn_point)
         fps = fps_index(rdn, ptc, ptc_fps_point)
         ptc = world_transform(ptc[fps])
         background_idx, cloth_idx = get_background_idx(rgb, ptc, fps)
         curv, curv_norm, norm = compute_surface_curvature(ptc)
         features[i] = torch.cat((curv, curv_norm.unsqueeze(1), norm), dim=1)
         ptcs[i] = ptc
         background_idxs.append(background_idx)
         cloth_idxs.append(cloth_idx)

         '''
         fig = plt.figure(figsize=(18, 12))
         ax0 = fig.add_subplot(231, projection='3d')
         ax1 = fig.add_subplot(232, projection='3d')
         ax2 = fig.add_subplot(233, projection='3d')
         ax3 = fig.add_subplot(234,projection='3d')
         ax4 = fig.add_subplot(235,projection='3d')
         ax5 = fig.add_subplot(236,projection='3d')

         x = ptc[:, 0]
         y = ptc[:, 1]
         z = ptc[:, 2]

         for ax, c in zip([ax0, ax1, ax2, ax3, ax4, ax5], [curv[:,0], curv[:,1], curv_norm, norm[:,0], norm[:,1], norm[:,2]]) :
            ax.scatter(x, y, z, c=c, cmap='viridis', s=50, depthshade=False)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(azim=90, elev=-90)

         plt.figure()
         plt.imshow(rgb)
         plt.show()

         fig = plt.figure(figsize=(10,5))
         ax = fig.add_subplot(121)
         ax.imshow(obs_list[i][:,:,:3])
         ax2 = fig.add_subplot(122)
         ax2.scatter(ptc[backround_idx,0], ptc[backround_idx,1], s=1, c='b')
         ax2.scatter(ptc[cloth_idx, 0], ptc[cloth_idx, 1], s= 1, c='r')
         plt.show()
         '''



      return ptcs, background_idxs, cloth_idxs, features

   def get_gaussian_probability(self, action_prob, pts, ptcs, background_idxs, cloth_idxs) :
      actions_gaussian = torch.zeros_like(action_prob)
      for i, action in tqdm(enumerate(action_prob), total=self.n_sample) :
         gt = action.reshape(25, 25)
         # Generate coordinate grids
         x = torch.arange(25).float()
         y = torch.arange(25).float()
         grid_x, grid_y = torch.meshgrid(x, y)

         # Get the coordinates of the element equal to 1
         coord_x, coord_y = torch.where(gt == 1)

         # Calculate the Gaussian distribution centered at the specified coordinates
         mean = torch.stack([coord_x.float(), coord_y.float()], dim=1)
         std = 3  # Standard deviation of the Gaussian distribution
         gaussian_tensor = torch.exp(-((grid_x - mean[:, 0])**2 + (grid_y - mean[:, 1])**2) / (2 * std**2))

         # Normalize the Gaussian tensor to have a maximum value of 1
         actions_gaussian[i,:] = (gaussian_tensor / torch.max(gaussian_tensor)).reshape(-1)
      
      heatmap_ptc = torch.zeros(self.n_sample, ptcs.shape[1])
      for i in range(self.n_sample):
         heatmap_ptc[i,background_idxs[i]] = 0.0

         for id in cloth_idxs[i] :
            pt = ptcs[i, id]
            dist = torch.linalg.norm((pts[i, :, :2] - pt[:2]), dim = 1)
            min = torch.where(dist == dist.min())[0]
            heatmap_ptc[i,id] = actions_gaussian[i,min]
         
         heatmap_ptc[i, cloth_idxs[i]] = heatmap_ptc[i, cloth_idxs[i]] * 0.8  + 0.2
      '''
      
      for i in range(self.n_sample) :


         fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

         heat =heatmap_ptc[i].cpu().numpy()
         ptc = self.ptcs[i].cpu().numpy()
         gaussian_tensor = actions_gaussian[i].cpu().numpy().reshape(25,25).transpose(1,0)



         # Define a custom colormap that goes from white to red
         cmap = mcolors.LinearSegmentedColormap.from_list("white_to_red", [(1, 1, 1), (1, 0, 0)], N=256)

         # Plot the first image (gt) on the left
         axs[0,0].imshow(gaussian_tensor, origin='lower', cmap=cmap, vmin=0, vmax=1)
         axs[0,0].set_title("Ground Truth (GT) ordered")

         axs[0,1].scatter(ptc[:,0], ptc[:,1], c=heat)
         axs[0,1].set_title("Prediction ordered")

         pt = self.pts[i].cpu().numpy()
         axs[1,0].scatter(pt[:,0], pt[:,1], c=actions_gaussian[i].cpu().numpy())
         axs[1,0].set_title("Mesh ordered")

         
         bkg   = self.ptcs[i, self.background_idxs[i]].cpu().numpy()
         cloth = self.ptcs[i, self.cloth_idxs[i]].cpu().numpy()

         axs[1,1].scatter(bkg[:, 0], bkg[:,1], s=1, c='b')
         axs[1,1].scatter(cloth[:, 0], cloth[:, 1], s= 1, c='r')
         plt.show()


         plt.tight_layout()
         
         print('Convert action to Gaussian')
         plt.show()

      '''
   
      return actions_gaussian, heatmap_ptc
   def set_obs_type(self, obs_type):
      if obs_type not in OBS_TYPE : raise ValueError(f'obs_type must be in {OBS_TYPE}')
      self.obs_type = obs_type

   def set_output_type(self, output_type):
      if output_type not in OUTPUT_TYPE : raise ValueError(f'output_type must be in {OUTPUT_TYPE}')
      self.output_type = output_type

   def to_device(self, device):
      self.device = device
      self.actions = self.actions.to(device)
      self.obs = self.obs.to(device)
      self.pts = self.pts.to(device)
      self.action_prob = self.action_prob.to(device)
      self.actions_gaussian = self.actions_gaussian.to(device)

   def shuffle_points(self) :
      for i, pts in enumerate(self.pts) :
         gaussian = self.actions_gaussian[i,:]
         prob = self.action_prob[i,:]
         permuted_indices = torch.randperm(pts.size()[0])
         # permute the batch
         self.pts[i,:]              = pts[permuted_indices,:]
         self.actions_gaussian[i,:] = gaussian[permuted_indices]
         self.action_prob[i,:]      = prob[permuted_indices]

      for i in range(self.ptcs.shape[0]) :
         permuted_indices = torch.randperm(self.ptcs[i].shape[0])
         # permute the batch
         self.ptcs[i,:] = self.ptcs[i, permuted_indices,:]
         self.heatmap_ptc[i,:] = self.heatmap_ptc[i, permuted_indices]

if __name__ == '__main__' :

   if GENERATE_LIST :
      
      print('Loading pkl...')
      with open(PKL_LOAD_PATH, 'rb') as f :
         data_pkl = pickle.load(f)   

      print('Converting Pkl to list...')
      action_list, obs_list, pts_list = [], [], []
      for episode in data_pkl :
         for i in range(len(episode['act'])) :
            action_list.append(torch.tensor(episode['act'][i]))
            obs_list.append(torch.tensor(episode['obs'][i]))
            pts_list.append(torch.tensor(episode['pts'][i]))

      print('Saving list...')
      torch.save({'action' : action_list,
                  'obs_list' : obs_list,
                  'pts_list' : pts_list},
                 PT_LIST_SAVE_PATH)
   else :
      print('Loading list from path ...')
      list_dict = torch.load(PT_LIST_SAVE_PATH)
      action_list = list_dict['action']
      obs_list = list_dict['obs_list']
      pts_list = list_dict['pts_list']

   if CREATE_DATASET :
      print('Creating dataset...')
      dataset = GymClothDataset(action_list, obs_list, pts_list)
      torch.save(dataset,DATASET_SAVE_PATH)
      del dataset

   if EXTEND_DATASET :
      dataset = torch.load(DATASET_SAVE_PATH)
      print('Initial lenght :', len(dataset))
      print('Total Data available :',len(action_list))
      dataset.extend(action_list[len(dataset):], obs_list[len(dataset):], pts_list[len(dataset):])
      print('Final lenght :', len(dataset))
      torch.save(dataset,DATASET_SAVE_PATH2)
      del dataset


   print('Testing data loader ...')
   loaded_dataset = torch.load(DATASET_SAVE_PATH)
   #loaded_dataset.set_obs_type('heatmap+ptc')
   #loaded_dataset.set_output_type('heatmap+ptc')
   loaded_dataset.to_device(torch.device('cpu'))
   loaded_dataset.shuffle_points()

   print(len(loaded_dataset))
   obs, gt = loaded_dataset[torch.randint(0, len(loaded_dataset), (1,))]

   pts = obs[0]
   ptc = obs[2]

   print(pts.shape)
   print(ptc.shape)
   print(gt[0].shape)
