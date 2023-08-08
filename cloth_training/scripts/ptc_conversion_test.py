import pickle
import torch
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Suppress the warning
from scipy.spatial.transform import Rotation as R


from scipy.spatial.distance import cdist

def farthest_point_sampling(point_cloud, n):
   start_time = time.time()
   sampled_points = np.zeros((n, point_cloud.shape[1]))
   sampled_indices = [np.random.randint(0, len(point_cloud))]
   for i in range(1, n):
      distances = cdist(point_cloud, point_cloud[sampled_indices])
      min_distances = np.min(distances, axis=1)
      new_index = np.argmax(min_distances)
      sampled_indices.append(new_index)
   sampled_points = point_cloud[sampled_indices]
   print(f'FPS took {time.time() - start_time} seconds')
   return sampled_points


def depth_to_point_cloud(depth, intrinsic_params):
   # get image dimensions
   height, width = depth.shape
   
   # unpack intrinsic parameters
   fx, _, cx, _, fy, cy, _, _, _ = intrinsic_params
   
   # create meshgrid of pixel coordinates
   x, y = np.meshgrid(np.arange(width), np.arange(height))
   
   # compute 3D coordinates of each pixel
   x3d = (x - cx) * depth / fx / RATIO
   y3d = (y - cy) * depth / fy / RATIO
   z3d = depth / RATIO
   
   point_cloud = np.stack([x3d, y3d, z3d], axis=-1)
   point_cloud = point_cloud.reshape(-1, 3)

   return point_cloud

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


def fps_point_cloud(point_cloud, n):
   # compute pairwise distances between points
   
   print('point cloud shape: ', point_cloud.shape)
   start_time = time.time()
   dists = np.sum((point_cloud[:, np.newaxis, :] - point_cloud[np.newaxis, :, :]) ** 2, axis=-1)
   
   # initialize list of selected points
   selected = [np.random.randint(len(point_cloud))]
   
   # select remaining points using FPS
   for i in range(1, n):
      dists_to_selected = dists[selected, :]
      min_dists = np.min(dists_to_selected, axis=0)
      new_point = np.argmax(min_dists)
      selected.append(new_point)
   
   print('fps time: ', time.time() - start_time)

   # return selected points
   fps_points = point_cloud[selected]
   return fps_points

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


   #for ax in [ax2,ax3] :
   #   ax.set_ylim3d(0,1)
   #   ax.set_xlim3d(0,1)

   ax1.set_title('Point Cloud')
   ax2.set_title('FPS Point Cloud')
   ax3.set_title('Random Points')

   plt.tight_layout()

   # Display the figure
   plt.show(block = block)


if __name__ == '__main__':
   save_to = 'ptc_test'
   dataset_path = '/home/kgalassi/code/cloth/gym-cloth-simulator/logs/evaluation-transformer-07-27-02-50-3-nosim-seed-None-obs-blender-depth-False-rgbd-True-tier3_epis_10.pkl'
   dataset_path = '/home/kgalassi/code/cloth/gym-cloth-simulator/logs/evaluation-transformer-07-25-22-37-seed-None-obs-blender-depth-False-rgbd-True-tier1_epis_10.pkl'
   dataset_path ='/home/kgalassi/code/cloth/gym-cloth-simulator/logs/data-seed-42-tier-3-seed-42-obs-blender-depth-False-rgbd-True-tier3_epis_10.pkl'

   BASE_PATH = '/home/kgalassi/code/cloth/cloth_training/cloth_training/scripts/results'
   os.makedirs(os.path.join(BASE_PATH, save_to), exist_ok=True)

   #THRESHOLD = 100
   RATIO = 1000.0

   FOCAL_LENGTH_MM = 40.0
   SENSOR_WIDTH_MM = 36.0
   IMAGE_WITDH_PIXELS = 224.0
   IMAGE_HEIGHT_PIXELS = 224.0
   CAMERA_TRANSFORM = [ 1.0, 0.0,  0.0, 0.5,
                        0.0, -1.0, 0.0, 0.5,
                        0.0, 0.0,  1.0, 2.0,
                        0.0, 0.0,  0.0, 1.0]
   FOCAL_LENGTH_PIXEL_X = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * IMAGE_WITDH_PIXELS
   FOCAL_LENGTH_PIXEL_Y = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * IMAGE_HEIGHT_PIXELS
   CX = IMAGE_WITDH_PIXELS / 2.0
   CY = IMAGE_HEIGHT_PIXELS / 2.0

   INTRINSIC_PARAMS = [FOCAL_LENGTH_PIXEL_X, 0, CX,
                     0, FOCAL_LENGTH_PIXEL_Y, CY,
                     0, 0, 1]


   with open(dataset_path, 'rb') as f:
      dataset = pickle.load(f)

   print('Dataset len loaded:' , len(dataset))

   coverage, variance_inv, num_steps = [], [], []

   for _ in range(3) :
      episode_no = np.random.randint(len(dataset))
      step_no = np.random.randint(len(dataset[episode_no]['act']))


      episode = dataset[episode_no]

      print('Plot episode: ', episode_no)
      print('Plot step: ', step_no)

      # Get Inputs
      act        = episode['act'][step_no]
      #act_policy = episode['act_policy'][step_no]
      rgb        = episode['obs'][step_no][:, :, :3]
      depth      = episode['obs'][step_no][:, :, 3]
      pts        = episode['pts'][step_no]
      dm         = episode['info'][step_no]['randomization']


      t_offset = np.array(dm['camera_pos']).reshape(1,3)
      t_offset = np.zeros((4,4))
      t_offset[:3,3] = np.array(dm['camera_pos']).reshape(3)
      t_offset[3,3] = 1


      t_rot = R.from_euler('xyz', dm['camera_deg'], degrees=True).as_matrix()
      t_rot = np.hstack([t_rot, np.zeros((3,1))])
      t_rot = np.vstack([t_rot, np.zeros((1,4))])
      t_rot[3,3] = 1


      print(dm.keys())
      print(dm['camera_deg'])
      print(dm['camera_pos'])
      print(dm['gval_depth'])
      depth = depth + dm['gval_depth'] - 50

      print('depth : ', depth.shape,' loaded')
      print(np.min(depth), np.max(depth))

      pts = np.array(pts)
      #change order of x,y in depth image
      ptc = depth_to_point_cloud(depth, INTRINSIC_PARAMS)
      print('pointcloud : ', ptc.shape,' loaded')

      x3d, y3d, z3d = ptc[..., 0].flatten(), ptc[..., 1].flatten(), ptc[..., 2].flatten()
      mask = np.logical_and(z3d>0.0, z3d < 0.18)
      x3d, y3d, z3d = x3d[mask], y3d[mask], z3d[mask]
      ptc = np.stack([x3d, y3d, z3d], axis=-1)



      print('pointcloud : ', ptc.shape,' loaded')
      #visualize_point_cloud(ptc, block=False)

      ptc = farthest_point_sampling(ptc, 3000)
      print('FPS OK')
      
      ones = np.ones((ptc.shape[0], 1))
      ptc_h = np.hstack([ptc, ones])
      t = np.array(CAMERA_TRANSFORM).reshape(4,4)
      t = t #* t_offset * t_rot

      ptc2 = np.empty((ptc_h.shape[0], 3))
      for i in range(ptc_h.shape[0]):
         outm = np.matmul(t, ptc_h[i])
         ptc2[i] = outm[:3]

      print('Transform OK')


      multiple_plot(ptc, ptc2, pts)

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
      
print('Coverage : ', np.mean(coverage))
print('Variance : ', np.mean(variance_inv))
print('Num Steps : ', np.mean(num_steps))

print('Coverage : ', np.std(coverage))
print('Variance : ', np.std(variance_inv))
print('Num Steps : ', np.std(num_steps))
