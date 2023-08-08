import torch
from torch.utils.data import Dataset
import torch.nn as nn

from tqdm import tqdm

import numpy as np
import os, pickle

from cloth_training.model.model_architecture.model_utils import depth_to_point_cloud, random_point_sampling, farthest_point_sampling
from cloth_training.model.model_architecture.model_utils import world_transform
from cloth_training.model.model_architecture.model_utils import compute_surface_curvature, depth_to_point_cloud_tensor,random_point_sampling_tensor
def set_seed(seed):
   if seed == None :
      print('Warning : seed is None')
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True


OBS_TYPE = ['rgb', 'd', 'rgbd', 'pts', 'heatmap', 'all', 'ptc', 'heatmap+ptc', 'vit']
OUTPUT_TYPE = ['prob', 'prob_gaussian', 'offset', 'heatmap','ptc','heatmap+ptc', 'vit']


PKL_LOAD_PATH     = '/home/kgalassi/code/cloth/gym-cloth-main/logs/data-2023-08-04-14-27-seed-43-tier-3-seed-43-obs-blender-depth-False-rgbd-True-tier3_epis_10.pkl'
PT_LIST_SAVE_PATH = '/home/kgalassi/code/cloth/cloth_training/cloth_training/dataset/ptc/pt_list.pt'
DATASET_SAVE_PATH = '/home/kgalassi/code/cloth/cloth_training/cloth_training/dataset/ptc/dataset.pt'
DATASET_SAVE_PATH2 = '/home/kgalassi/code/cloth/cloth_training/cloth_training/dataset/ptc/dataset2.pt'
GENERATE_LIST = True
CREATE_DATASET = False
EXTEND_DATASET = True

class GymClothDataset(Dataset):
   def __init__(self, action_list, obs_list, pts_list,
                obs_type = 'rgb', # 'rgb','d', pts', 'all'
                output_type = 'prob', #'prob, 'prob_gaussian', 'offset'
                pts_random = False,
                seed = None,
                ptc_rdn_point = 10000,
                ptc_fps_point = 3000):
      
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
      
      num_sample = self.actions.shape[0]

      # Find the match between grasp point and cloth point
      act = (self.actions[:,:2] / 2.0) + 0.5
      distances2 = torch.norm(self.pts[:,:,:2]  - act[:,:2].unsqueeze(1), dim=2)
      min_distance_idx2 = torch.argmin(distances2, dim=1)
      self.action_prob = torch.zeros((num_sample, self.pts.shape[1]))
      self.action_prob[torch.arange(num_sample), min_distance_idx2] = 1
      

      ptcs = torch.empty(num_sample, ptc_fps_point, 3)
      ptcr = torch.empty(num_sample, ptc_rdn_point, 3)
      features = torch.empty(num_sample, ptc_fps_point, 6)


      min = 1.4 - torch.max(self.pts[:, :, 2], dim=1)[0].unsqueeze(1).float()
      max = 1.4 - torch.min(self.pts[:, :, 2], dim=1)[0].unsqueeze(1).float()
      pt = depth_to_point_cloud_tensor(self.obs[:,:,:,3], max=max, min=min)
      ptcr = random_point_sampling_tensor(pt, ptc_rdn_point)

      for i in tqdm(range(num_sample), total=num_sample):
         #depth = self.obs[i,:,:,3]
         #min = 1.4 - torch.max(self.pts[i, :,2]).float()
         #max = 1.4 - torch.min(self.pts[i, :,2]).float()
         #ptc = depth_to_point_cloud(depth, max=max, min=min)
         #ptc  = depth_to_point_cloud(self.obs[i,:,:,3], max=max[i], min=min[i])
         #ptc = random_point_sampling(ptc, ptc_rdn_point)
         ptc = farthest_point_sampling(ptcr[i], ptc_fps_point)
         ptc = world_transform(ptc)
         curv, curv_norm, norm = compute_surface_curvature(ptc)
         features[i] = torch.cat((curv, curv_norm.unsqueeze(1), norm), dim=1)
         ptcs[i] = ptc

      self.ptc = ptcs
      self.features = features

      print('Dataset length : ', self.action_prob.shape[0])
      print('Action prob. shape : ', self.action_prob.shape)
      print('Action shape : ', self.actions.shape)
      print('Obs. shape : ', self.obs.shape)
      print('Pts. shape : ', self.pts.shape)
      print('Ptc. shape : ', self.ptc.shape)
      print('Features shape : ', self.features.shape)
      print('+++++++++++++')

      # Create a gaussian distribution centered at the action
      self.actions_gaussian = torch.zeros_like(self.action_prob)
      for i, action in tqdm(enumerate(self.action_prob), total=self.action_prob.shape[0]) :
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
         self.actions_gaussian[i,:] = (gaussian_tensor / torch.max(gaussian_tensor)).reshape(-1)
      print('Convert action to Gaussian')

   def __len__(self):
      return self.obs.shape[0]
   
   def __getitem__(self, index):

      if self.obs_type == 'rgb' :
         obs = self.obs[index,:,:,:3]
      elif self.obs_type == 'pts' :
         obs = self.pts[index]
      elif self.obs_type == 'heatmap' :
         obs = (self.pts[index], self.actions_gaussian[index]) # (Nx3, N)
      elif self.obs_type == 'ptc' :
         obs = self.ptc[index] #Nx3
      elif self.obs_type == 'heatmap+ptc':
         obs = (self.pts[index], self.actions_gaussian[index], self.ptc[index]) # (Nx3, N, Nx3)
      elif self.obs_type == 'vit' :
         obs = self.obs[index][:,:,:3].reshape(3,224, 224) # (3,224,224)
      elif self.obs_type == 'feature' :#
         obs = (self.pts[index], self.actions_gaussian[index], self.ptc[index], self.features) # (Nx3, N, Nx3)
      else :
         obs = (self.obs[index], self.pts[index])


      if self.output_type == 'prob' :
         out = (self.action_prob[index], self.actions[index,2:])
      elif self.output_type == 'prob_gaussian' :
         out = (self.actions_gaussian[index], self.actions[index,2:])
      elif self.output_type == 'heatmap' :
         out = (self.actions_gaussian[index], self.actions[index]) #(N, 4)
      elif self.output_type == 'ptc' :
         out = self.pts[index] # Nx3
      elif self.output_type == 'heatmap+ptc' :
         out = (self.actions_gaussian[index], self.actions[index], self.pts[index])
      elif self.output_type == 'vit' :
         out = self.pts[index]
      else :
         out = (self.actions[index,:2], self.actions[index,2:])
      
      return obs, out


   def extend(self, action_list, obs_list, pts_list, ptc_fps_point=3000, ptc_rdn_point=10000) :
      actions = torch.stack(action_list)
      obs     = torch.stack(obs_list)
      pts     = torch.stack(pts_list)
      
      num_sample = actions.shape[0]

      # Find the match between grasp point and cloth point
      act = (actions[:,:2] / 2.0) + 0.5
      distances2 = torch.norm(pts[:,:,:2]  - act[:,:2].unsqueeze(1), dim=2)
      min_distance_idx2 = torch.argmin(distances2, dim=1)
      action_prob = torch.zeros((num_sample, pts.shape[1]))
      action_prob[torch.arange(num_sample), min_distance_idx2] = 1
      
      ptcr = torch.empty(num_sample, ptc_rdn_point, 3)
      ptcs = torch.empty(num_sample, ptc_fps_point, 3)
      features = torch.empty(num_sample, ptc_fps_point, 6)


      min = 1.4 - torch.max(pts[:, :, 2], dim=1)[0].unsqueeze(1).float()
      max = 1.4 - torch.min(pts[:, :, 2], dim=1)[0].unsqueeze(1).float()
      pt = depth_to_point_cloud_tensor(obs[:,:,:,3], max=max, min=min)
      ptcr = random_point_sampling_tensor(pt, ptc_rdn_point)

      for i in tqdm(range(num_sample), total=num_sample):
         ptc = farthest_point_sampling(ptcr[i], ptc_fps_point)
         ptc = world_transform(ptc)
         curv, curv_norm, norm = compute_surface_curvature(ptc)
         features[i] = torch.cat((curv, curv_norm.unsqueeze(1), norm), dim=1)
         ptcs[i] = ptc

      ptc = ptcs
      features = features


      # Create a gaussian distribution centered at the action
      actions_gaussian = torch.zeros_like(action_prob)
      for i, action in tqdm(enumerate(action_prob), total=action_prob.shape[0]) :
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
      print('Convert action to Gaussian')



      self.actions          = torch.cat((self.actions, actions), dim=0)
      self.actions_gaussian = torch.cat((self.actions_gaussian, actions_gaussian), dim=0)
      self.obs              = torch.cat((self.obs, obs), dim=0)
      self.pts              = torch.cat((self.pts, pts), dim=0)
      self.action_prob      = torch.cat((self.action_prob, action_prob), dim=0)
      self.ptc              = torch.cat((self.ptc, ptc), dim=0)
      self.features         = torch.cat((self.features, features), dim=0)
      

      #Add new data to database


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

      for i, ptc in enumerate(self.ptc) :
         permuted_indices = torch.randperm(ptc.size()[0])
         # permute the batch
         self.ptc[i,:] = ptc[permuted_indices,:]


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
   loaded_dataset.set_obs_type('heatmap+ptc')
   loaded_dataset.set_output_type('heatmap+ptc')
   loaded_dataset.to_device(torch.device('cpu'))
   loaded_dataset.shuffle_points()

   print(len(loaded_dataset))
   obs, gt = loaded_dataset[torch.randint(0, len(loaded_dataset), (1,))]

   pts = obs[0]
   ptc = obs[2]

   print(pts.shape)
   print(ptc.shape)
   print(gt[0].shape)
