import torch
import os
import time

from cloth_training.model.model_architecture.dataset_gen import GymClothDataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from scipy.spatial.distance import cdist

def visualize_point_cloud(point_cloud, block=True):
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   
   # extract x, y, z coordinates from point cloud
   x = point_cloud[..., 0].flatten()
   y = point_cloud[..., 1].flatten()
   z = point_cloud[..., 2].flatten()
   
   # plot point cloud
   ax.scatter(x, y, z, s=1)
   
   # set axis labels
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   
   # set axis limits
   ax.set_xlim3d(np.min(x), np.max(x))
   ax.set_ylim3d(np.min(y), np.max(y))
   ax.set_zlim3d(np.min(z), np.max(z))
   
   ax.view_init(elev=90, azim=0)  # Set elevation to 90 degrees to invert the z-axis


   plt.show(block=block)


def show_depth_image(depth, block=True):
   plt.figure()
   plt.imshow(depth, cmap='coolwarm')
   plt.show(block=block)


def comparison_plot(fps_pointcloud, pts, block=True) :

   fig = plt.figure(figsize=(20, 10))


   ax1 = fig.add_subplot(121, projection='3d')  # 1 row, 3 columns, subplot 1
   ax2 = fig.add_subplot(122, projection='3d')  # 1 row, 3 columns, subplot 2

   for ax, ptc,c in zip([ax1,ax2], [fps_pointcloud, pts], ['r', 'g', 'b']) :
      x = ptc[..., 0].flatten()
      y = ptc[..., 1].flatten()
      z = ptc[..., 2].flatten()
      ax.scatter(x, y, z, s=10, c=c)
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      ax.set_xlim3d(0.0, 1.0)
      ax.set_ylim3d(0.0, 1.0)
      ax.set_zlim3d(np.min(z), np.max(z))
      ax.view_init(elev=90, azim=-90)  # Set elevation to 90 degrees to invert the z-axis

   ax1.set_title('FPS Point Cloud')
   ax2.set_title('Random Points')

   plt.tight_layout()

   # Display the figure
   plt.show(block=block)


def multiple_plot(pointcloud, fps_pointcloud, pts, block=True) :

   fig = plt.figure(figsize=(30, 10))


   ax1 = fig.add_subplot(131, projection='3d')  # 1 row, 3 columns, subplot 1
   ax2 = fig.add_subplot(132, projection='3d')  # 1 row, 3 columns, subplot 2
   ax3 = fig.add_subplot(133, projection='3d')  # 1 row, 3 columns, subplot 3

   for ax, ptc,c in zip([ax1,ax2,ax3], [pointcloud, fps_pointcloud, pts], ['r', 'g', 'b']) :
      x = ptc[..., 0].flatten()
      y = ptc[..., 1].flatten()
      z = ptc[..., 2].flatten()
      ax.scatter(x, y, z, s=10, c=c)
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')

      ax.view_init(elev=90, azim=-90)  # Set elevation to 90 degrees to invert the z-axis


   for ax in [ax2,ax3] :
      ax.set_ylim3d(0,1)
      ax.set_xlim3d(0,1)

   ax1.set_title('Point Cloud')
   ax2.set_title('FPS Point Cloud')
   ax3.set_title('Random Points')

   plt.tight_layout()

   # Display the figure
   plt.show(block = block)


if __name__ == '__main__':
      
   save_to = 'ptc_test'
   dataset_path = '/home/kgalassi/code/cloth/cloth_training/cloth_training/dagger/pull/pull_dataset3.pt'
   BASE_PATH = '/home/kgalassi/code/cloth/cloth_training/cloth_training/scripts/results'
   os.makedirs(os.path.join(BASE_PATH, save_to), exist_ok=True)
   #THRESHOLD = 100
   RATIO = 100.0
   FOCAL_LENGTH_MM = 40.0
   SENSOR_WIDTH_MM = 36.0
   IMAGE_WITDH_PIXELS = 224.0
   IMAGE_HEIGHT_PIXELS = 224.0
   CAMERA_TRANSFORM = [ 1.0, 0.0,  0.0, 0.5,
                        0.0,  -1.0,  0.0, 0.5,
                        0.0,  0.0, -1.0, 1.45,
                        0.0,  0.0,  0.0, 1.0]

   FOCAL_LENGTH_PIXEL_X = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * IMAGE_WITDH_PIXELS
   FOCAL_LENGTH_PIXEL_Y = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * IMAGE_HEIGHT_PIXELS
   CX = IMAGE_WITDH_PIXELS / 2.0
   CY = IMAGE_HEIGHT_PIXELS / 2.0

   INTRINSIC_PARAMS = [FOCAL_LENGTH_PIXEL_X, 0, CX,
                     0, FOCAL_LENGTH_PIXEL_Y, CY,
                     0, 0, 1]

   
   dataset = torch.load(dataset_path)
   dataset.set_obs_type('heatmap+ptc')
   dataset.set_output_type('heatmap+ptc')
   dataset.to_device(torch.device('cpu'))
   dataset.shuffle_points()

   print('Dataset len loaded:' , len(dataset))
   for _ in range(3) :

      obs, gt = dataset[torch.randint(0, len(dataset), (1,))]

      pts = obs[0]
      ptc = obs[2]

      ptc = ptc.numpy()
      pts = pts.numpy()
      comparison_plot(ptc, pts)
      #multiple_plot(ptc, ptc2, pts)

      #comparison_plot(ptc, pts)

      #visualize_point_cloud(ptc, block=False)
      #visualize_point_cloud(ptc2, block=False)
      #show_depth_image(depth)

   exit()
      


   for i,episode in tqdm(enumerate(dataset), total= len(dataset),desc='Episodes'):
      totale_episode = len(episode['act'])
      for j in range(totale_episode):

         # Get Inputs
         act   = episode['act'][j]
         act_policy = episode['act_policy'][j]
         rgb   = episode['obs'][j][:, :, :3]
         depth = episode['obs'][j][:, :, 3]
         
         pts   = episode['pts'][j]
         

         ptc = depth_to_point_cloud(depth, INTRINSIC_PARAMS)
         
         
         visualize_point_cloud(ptc, block=False)
         show_depth_image(depth)
         
         exit()
      
