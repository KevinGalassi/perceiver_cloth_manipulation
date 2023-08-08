import pickle
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import matplotlib.colors as mcolors
from matplotlib.patches import Arrow



def get_matplotlib_heatmap(heat, gaussian_tensor, permuted_indices):

   plt.clf()
   fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

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


   for ax in axs.flat:
      ax.set_xlim(0, 25)
      ax.set_ylim(0, 25) 
      ax.set_xticks(range(0, 30, 5))
      ax.set_yticks(range(0, 30, 5))

   plt.tight_layout()

   return fig

def get_matplotlib_action(pts, pt_id, gt_id, a, a_gt) :
   pts_numpy = pts.reshape(-1,3).detach().cpu().squeeze(0).numpy()
   a = a.reshape(2).detach().cpu().squeeze(0).numpy()
   a_gt = a_gt.reshape(2).detach().cpu().squeeze(0).numpy()

   plt.clf()
   fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(40, 20))

   pt_taken = np.reshape(pts_numpy[pt_id, :], (3))
   pt_gt    = np.reshape(pts_numpy[gt_id, :], (3))

   axs[0].scatter(pts_numpy[:,0], pts_numpy[:,1], c='b', s=PTS_DIM)
   axs[0].scatter(pt_gt[0], pt_gt[1], c='r', s=PT_DIM)
   #axs[0].plot((pt_gt[0], pt_gt[0] + a_gt[-2]),(pt_gt[1], pt_gt[1] + a_gt[-1]), c='g')
   axs[0].arrow(x=pt_gt[0],y= pt_gt[1], dx=a_gt[-2], dy=a_gt[-1], width=ARROW_WIDTH)


   axs[1].scatter(pts_numpy[:,0], pts_numpy[:,1], c='b', s=PTS_DIM)
   axs[1].scatter(pt_taken[0], pt_taken[1], c='r', s=PT_DIM)
   #axs[0].plot((pt_taken[0], pt_taken[0] + a[-2]),(pt_taken[1], pt_taken[1] + a[-1]), c='g')
   axs[1].arrow(x=pt_taken[0],y= pt_taken[1], dx=a[-2], dy=a[-1], width=ARROW_WIDTH)
   
   axs[0].set_title("Action Ground Truth")
   axs[1].set_title("Action Prediction")

   # Create a custom legend with an arrow
   legend_labels = ['points', 'grasp point', 'action vector']
   legend_elements = [
      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=LEGEND_SIZE, label=legend_labels[0]),
      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=LEGEND_SIZE, label=legend_labels[1]),
      Arrow(0, 0, 0.0001, 0, width=0.0001,label=legend_labels[2])
   ]
   
   axs[0].legend(handles=legend_elements)
   axs[1].legend(handles=legend_elements)
   plt.tight_layout()

   return fig


def get_matplotlib_single_action(pts, pt_id, a) :
   pts_numpy = pts.reshape(-1,3).detach().cpu().squeeze(0).numpy()
   a = a.reshape(2).detach().cpu().squeeze(0).numpy()

   plt.clf()
   fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))

   pt_taken = np.reshape(pts_numpy[pt_id, :], (3))

   ax.scatter(pts_numpy[:,0], pts_numpy[:,1], c='b', s=PTS_DIM)
   ax.arrow(x=pt_taken[0],y= pt_taken[1], dx=a[-2], dy=a[-1], width=ARROW_WIDTH)
   ax.scatter(pt_taken[0], pt_taken[1], c='r', s=PT_DIM)
   ax.set_title("Action Prediction")

   # Create a custom legend with an arrow
   legend_labels = ['points', 'grasp point', 'action vector']
   legend_elements = [
      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=LEGEND_SIZE, label=legend_labels[0]),
      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=LEGEND_SIZE, label=legend_labels[1]),
      Arrow(0, 0, 0.0001, 0, width=0.0001,label=legend_labels[2])
   ]

   ax.legend(handles=legend_elements)

   plt.tight_layout()

   return fig


def get_ground_truth_id(action, pts) :
   act = (action / 2.0) + 0.5
   distances2 = torch.norm(pts[:,:2]  - act, dim=-1)
   min_distance_idx2 = torch.argmin(distances2, dim=-1)
   action_prob = torch.zeros(pts.shape[0])
   action_prob[min_distance_idx2] = 1

   return action_prob
   
def get_gaussian_distrib(action_prob) :
   '''
   action_prob  | Size : 3
   pts          | Size : nx3  
   action       | Size : N
   '''

   # Create a gaussian distribution centered at the action
   gt = action_prob.reshape(25, 25)
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
   actions_gaussian = (gaussian_tensor / torch.max(gaussian_tensor)).reshape(-1)

   return actions_gaussian

   

if __name__ == '__main__':
   plt.rcParams['font.size'] = 30
   PTS_DIM = 150
   PT_DIM = 400
   ARROW_WIDTH = 0.01
   LEGEND_SIZE = 20
   SAVE_IMG = True

   save_to = '07-27-02-50-3'
   dataset_path = '/home/kgalassi/code/cloth/gym-cloth-simulator/logs/evaluation-transformer-07-27-02-50-3-seed-None-obs-blender-depth-False-rgbd-True-tier3_epis_10.pkl'
   BASE_PATH = '/home/kgalassi/code/cloth/cloth_training/cloth_training/scripts/results'
   os.makedirs(os.path.join(BASE_PATH, save_to), exist_ok=True)

   output = ['rgb', 'depth', 'heatmap', 'single','actionplot']
   for out in output : os.makedirs(os.path.join(BASE_PATH, save_to, out), exist_ok=True)

   with open(dataset_path, 'rb') as f:
      dataset = pickle.load(f)

   print(len(dataset))

   coverage, variance_inv, num_steps = [], [], []
   


   for i,episode in tqdm(enumerate(dataset), total= len(dataset),desc='Episodes'):
      totale_episode = len(episode['act'])
      for j in range(totale_episode):

         # Get Inputs
         act   = episode['act'][j]
         act_policy = episode['act_policy'][j]
         rgb   = episode['obs'][j][:, :, :3]
         depth = episode['obs'][j][:, :, 3]
         
         pts   = episode['pts'][j]
         
         heat  = episode['network_info'][j]['heat']
         indices = episode['network_info'][j]['indices']
         action = episode['network_info'][j]['action']
         prob   = episode['network_info'][j]['prob']

         # Convert to tensor
         act_policy = torch.tensor(act_policy)
         pts = torch.tensor(pts)

         # Create Missing data
         gt_prob = get_ground_truth_id(act_policy[:2], pts)
         gt_action = act_policy[2:]
         gt_id = torch.argmax(gt_prob, dim=-1)
         gaussian_tensor = get_gaussian_distrib(gt_prob)
         pt_id = torch.argmax(prob[0, torch.argsort(indices)], dim=-1)

         # Create Images
         if SAVE_IMG :
            heatmap = get_matplotlib_heatmap(heat, gaussian_tensor, indices)
            heatmap.savefig(os.path.join(BASE_PATH, save_to, 'heatmap', f'heatmap_{i}_{j}.eps'))
            #heatmap.savefig(os.path.join(BASE_PATH, save_to, 'heatmap', f'heatmap_{i}_{j}.png'))
            
            actionplot  = get_matplotlib_action(pts, pt_id, gt_id, action, gt_action)
            actionplot.savefig(os.path.join(BASE_PATH, save_to, 'actionplot', f'actionplot_{i}_{j}.eps'))
            #actionplot.savefig(os.path.join(BASE_PATH, save_to, 'actionplot', f'actionplot_{i}_{j}.png'))

            single = get_matplotlib_single_action(pts, pt_id, action)
            single.savefig(os.path.join(BASE_PATH, save_to, 'single', f'single_{i}_{j}.eps'))
            #single.savefig(os.path.join(BASE_PATH, save_to, 'single', f'single_{i}_{j}.png'))

            Image.fromarray(rgb).save(os.path.join(BASE_PATH, save_to, 'rgb', f'rgb_{i}_{j}.png'))
            Image.fromarray(depth).save(os.path.join(BASE_PATH, save_to, 'depth', f'depth_{i}_{j}.png'))

      coverage.append(episode['info'][-1]['actual_coverage'])
      variance_inv.append(episode['info'][-1]['variance_inv'])
      num_steps.append(totale_episode)
      
print('Coverage : ', np.mean(coverage))
print('Variance : ', np.mean(variance_inv))
print('Num Steps : ', np.mean(num_steps))

print('Coverage : ', np.std(coverage))
print('Variance : ', np.std(variance_inv))
print('Num Steps : ', np.std(num_steps))
