import torch
from torch.utils.data import Dataset
import torch.nn as nn

from tqdm import tqdm

import numpy as np
import os, pickle

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
DATASET_SAVE_PATH = '/home/kgalassi/code/cloth/cloth_training/cloth_training/dataset/dagger'
generate_list = False
create_dataset = True

NOME_DATASET = 'dataset3'
NOME_EP_LIST = 'dataset3_list'
#THRESHOLD = 100
RATIO = 100.0

FOCAL_LENGTH_MM = 40.0
SENSOR_WIDTH_MM = 36.0
IMAGE_WITDH_PIXELS = 224.0
IMAGE_HEIGHT_PIXELS = 224.0

FOCAL_LENGTH_PIXEL_X = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * IMAGE_WITDH_PIXELS
FOCAL_LENGTH_PIXEL_Y = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * IMAGE_HEIGHT_PIXELS
CX = IMAGE_WITDH_PIXELS / 2
CY = IMAGE_HEIGHT_PIXELS / 2
INTRINSIC_PARAMS = [FOCAL_LENGTH_PIXEL_X, 0, CX,
                    0, FOCAL_LENGTH_PIXEL_Y, CY,
                    0, 0, 1]
CAMERA_TRANSFORM = [ 1.0,  0.0,  0.0, 0.5,
                     0.0, -1.0,  0.0, 0.5,
                     0.0,  0.0, -1.0, 1.45,
                     0.0,  0.0,  0.0, 1.0]


def euclidean_distance(point_cloud):
    """
    Calculate pairwise Euclidean distances between all points in the point cloud.

    Args:
        point_cloud (torch.Tensor): Point cloud tensor of shape (N, 3) where N is the number of points.

    Returns:
        torch.Tensor: Pairwise Euclidean distances matrix of shape (N, N).
    """
    point_cloud_squared = torch.sum(point_cloud ** 2, dim=1, keepdim=True)
    pairwise_distances = -2 * torch.matmul(point_cloud, point_cloud.t()) + point_cloud_squared + point_cloud_squared.t()
    pairwise_distances = torch.sqrt(torch.max(pairwise_distances, torch.zeros_like(pairwise_distances)))
    return pairwise_distances


def farthest_point_sampling(point_cloud, num_samples):
   """
   Farthest Point Sampling algorithm for point clouds.

   Args:
      point_cloud (torch.Tensor): Point cloud tensor of shape (N, 3) where N is the number of points.
      num_samples (int): Number of points to sample.

   Returns:
      torch.Tensor: Indices of the sampled points.
   """

   num_points = point_cloud.shape[0]
   sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=point_cloud.device)
   distances = torch.full((num_points,), float('inf'), device=point_cloud.device)
   farthest_idx = torch.randint(0, num_points, (1,), dtype=torch.long, device=point_cloud.device)
   for i in range(num_samples):
      sampled_indices[i] = farthest_idx
      farthest_point = point_cloud[farthest_idx]
      dist_to_farthest = torch.norm(point_cloud - farthest_point, dim=1)
   
   
      mask = dist_to_farthest < distances
   

      distances[mask] = dist_to_farthest[mask]
      farthest_idx = torch.argmax(distances)


   return point_cloud[sampled_indices]

class GymClothDataset(Dataset):
   def __init__(self, action_list, obs_list, pts_list,
                obs_type = 'rgb', # 'rgb','d', pts', 'all'
                output_type = 'prob', #'prob, 'prob_gaussian', 'offset'
                pts_random = False,
                seed = None,
                ptc_sample='random'):
      

      #action_list = action_list[:2000]
      #obs_list = obs_list[:2000]
      #pts_list = pts_list[:2000]



      #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.pts_random = pts_random
      set_seed(seed) if seed is not None else None
      
      if obs_type not in OBS_TYPE : raise ValueError(f'obs_type must be in {OBS_TYPE}')
      self.obs_type = obs_type         

      if output_type not in OUTPUT_TYPE : raise ValueError(f'output_type must be in {OUTPUT_TYPE}')
      self.output_type = output_type

      #stack list of tensor
      self.actions = torch.stack(action_list)
      self.obs = torch.stack(obs_list)
      self.pts = torch.stack(pts_list)
      
      num_sample = self.actions.shape[0]

      # points are ordere by episode. Let's shuffle them
      #perm_indices = torch.randperm(self.actions.shape[0])
      #self.actions = self.actions[perm_indices]
      #self.obs = self.obs[perm_indices]
      #self.pts = self.pts[perm_indices]

      # Find the match between grasp point and cloth point
      act = (self.actions[:,:2] / 2.0) + 0.5
      distances2 = torch.norm(self.pts[:,:,:2]  - act[:,:2].unsqueeze(1), dim=2)
      min_distance_idx2 = torch.argmin(distances2, dim=1)
      self.action_prob = torch.zeros((num_sample, self.pts.shape[1]))
      self.action_prob[torch.arange(num_sample), min_distance_idx2] = 1
      

      print('Start length : ', self.action_prob.shape[0])


      #Create PointCloud
      self.height, self.width = self.obs.shape[1], self.obs.shape[2]
      fx, _, cx, _, fy, cy, _, _, _ = INTRINSIC_PARAMS
      x,y = torch.meshgrid(torch.arange(self.width), torch.arange(self.height))



      min_ptc_points = 3000
      ptc = []
      select_data = torch.empty(num_sample, dtype=bool)

      for i in tqdm(range(num_sample), total=num_sample):
         x3d = (x - cx) * self.obs[i,:,:,3] / fx / RATIO
         y3d = (y - cy) * self.obs[i,:,:,3] / fy / RATIO
         z3d = self.obs[i,:,:,3] / RATIO

         x3d = x3d.reshape(x3d.shape[0], -1)
         y3d = y3d.reshape(y3d.shape[0], -1)
         z3d = z3d.reshape(z3d.shape[0], -1)

         mask = torch.logical_and(z3d < 1.52, z3d > 0.5)
         x3d, y3d, z3d = x3d[mask], y3d[mask], z3d[mask]

         if z3d.shape[0] < min_ptc_points :
            select_data[i] = False
            continue

         point_cloud = torch.stack([x3d, y3d, z3d], axis=-1)

         ptc.append(farthest_point_sampling(point_cloud, min_ptc_points))


      ones = torch.ones((num_sample, min_ptc_points,1))
      ptc = torch.stack(ptc).reshape(-1, min_ptc_points, 3)
      ptc_h = torch.cat([ptc, ones], dim =-1)

      t = torch.tensor(CAMERA_TRANSFORM).reshape(4,4)
      for i in range(num_sample):
         for j in range(min_ptc_points):
            ptc[i,j] = torch.matmul(t, ptc_h[i,j])[:3]

      self.ptc = ptc
      self.action_prob = self.action_prob[select_data]
      self.actions = self.actions[select_data]
      self.obs = self.obs[select_data]
      self.pts = self.pts[select_data]

      print('Final length : ', self.action_prob.shape[0])
      print(self.action_prob.shape)
      print(self.actions.shape)
      print(self.obs.shape)
      print(self.ptc.shape)
      print(self.pts.shape)
      print('+++++++++++++')

      # Create a gaussian distribution centered at the action
      self.actions_gaussian = torch.zeros_like(self.action_prob)
      for i, action in enumerate(self.action_prob) :

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


   def __len__(self):
      return self.obs.shape[0]
   
   def __getitem__(self, index):

      if self.obs_type == 'rgb' :
         obs = self.obs[index,:,:,:3]
      elif self.obs_type == 'd' :
         obs = self.obs[index,:,:,3:]
      elif self.obs_type == 'rgbd' :
         obs = self.obs[index]
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

   oracle_path = '/home/kgalassi/code/cloth/cloth_training/cloth_training/dataset/raw'

   baselines = ['wrinkle', 'reveal', 'pull']
   #tiers = ['tier1', 'tier2', 'tier3']
   tiers = ['tier1']

   select_baseline = 'pull'
      

   if generate_list :

      # list file name in oracle_path
      oracle_file_list = [file for file in os.listdir(oracle_path) if file.endswith('.pkl')]

      episodes = {}
      for baseline in baselines:
         episodes[baseline] = {}
         for tier in tiers:
            episodes[baseline][tier] = []

      # Creation of episodes dictionary
      for path in oracle_file_list:
         filename = os.path.basename(path)
         filename_parts = filename.split('_')[0]   

         # Load datas

         # Find baseline and tier
         for tier_id in tiers :
            if tier_id in filename_parts:
               tier = tier_id
               break
         print('tier : ', tier)

         for baseline in baselines:
            print('baseline : ', baseline)
            if baseline in filename_parts:
               with open(os.path.join(oracle_path, path), 'rb') as f:
                  data = pickle.load(f)

               for episode in data:
                  episodes[baseline][tier].append(episode)
               print('number of episodes loaded : ', len(data))

               n_steps = 0
               for episode in data:
                  n_steps += len(episode['act'])
               print('number of steps loaded: ', n_steps)
               break

      episode_list = []
      for key in episodes[select_baseline]:
         for ep in episodes[select_baseline][key]:
            episode_list.append(ep)


      pts_list, obs_list, action_list = [], [], []
      for episode in episode_list:
         for i in range(len(episode['act'])) :
            action_list.append(torch.tensor(episode['act'][i]))
            obs_list.append(torch.tensor(episode['obs'][i]))
            pts_list.append(torch.tensor(episode['pts'][i]))

      torch.save({'action' : action_list,
                  'obs_list' : obs_list,
                  'pts_list' : pts_list},
                 os.path.join(DATASET_SAVE_PATH, select_baseline, select_baseline + '_' + NOME_DATASET + '_list.pt'))

   else :
      print('Loading list')
      list_dict = torch.load(os.path.join(DATASET_SAVE_PATH, select_baseline, select_baseline + '_' + NOME_EP_LIST+'.pt'))
      action_list = list_dict['action']
      obs_list = list_dict['obs_list']
      pts_list = list_dict['pts_list']

   if create_dataset :
      dataset = GymClothDataset(action_list, obs_list, pts_list)

      os.makedirs(os.path.join(DATASET_SAVE_PATH, select_baseline), exist_ok=True)
      print('Saving dataset in ', os.path.join(DATASET_SAVE_PATH, select_baseline, select_baseline + '_' + NOME_DATASET + '.pt'))
      torch.save(dataset, os.path.join(DATASET_SAVE_PATH, select_baseline, select_baseline + '_' + NOME_DATASET + '.pt'))

      print('Len of dataset : ', len(dataset))

      print('Test Loading dataset')
      del dataset

   print('Testing data load')
   loaded_dataset = torch.load(os.path.join(DATASET_SAVE_PATH, select_baseline, select_baseline+ '_' + NOME_DATASET + '.pt'))
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

   exit()
   
   print('Testing dataset')
   #OBS_TYPE = ['rgb', 'd', 'rgbd', 'pts', 'all']
   #OUTPUT_TYPE = ['prob', 'prob_gaussian', 'offset']

   loaded_dataset.set_obs_type('all')
   loaded_dataset.set_output_type('offset')

   # get random item from dataset
   obs, action = loaded_dataset[5000]

   print(type(obs))
   print(type(action))


   print(obs[0].shape)
   print(obs[1].shape)

   print(action[0].shape)
   print(action[1].shape)

   print('dataset lenght : ', len(loaded_dataset))

   exit()



'''

if False:
# get random item from dataset
obs, pts, action, act_prob = dataset[5000]

# create two plot with pts
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)


ax1.scatter(pts[:,0].cpu().numpy(), pts[:,1].cpu().numpy())
ax1.scatter(pts[act_prob,0].cpu().numpy(), pts[act_prob,1].cpu().numpy(), c='r')

plt.savefig('test.png')
#Plot Gaussian 
   for i in range(10) :
   obs, pts, action, act_prob = dataset[i]

   gt = act_prob.reshape(25, 25)
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
   gaussian_tensor = gaussian_tensor / torch.max(gaussian_tensor)

   # Convert the tensor to a NumPy array
   gaussian_array = gaussian_tensor.numpy()

   # Plot the tensor as an image
   plt.clf()
   plt.imshow(gaussian_array, cmap='hot', interpolation='nearest')
   plt.colorbar()
   plt.title('Gaussian Tensor')

   plt.savefig('test' + str(i) + '.png')
'''