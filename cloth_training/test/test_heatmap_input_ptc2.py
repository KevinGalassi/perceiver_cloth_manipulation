



import torch
from cloth_training.model.model_architecture.dataset_gen import GymClothDataset
from cloth_training.model.perceiver_heatmap import HeatPerceiver

import os

write_log = True
save_model = True

run_id = '07-27-02-50'
model_path = './model/saved_model/heat_map'
log_path = './model/logs/heat_map_input_ptc'


model = torch.load(os.path.join(model_path, run_id + '.pth'))
os.makedirs(os.path.join(log_path, run_id), exist_ok=True)
hparams = model['init_params']
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
   plt.close()

def poitnclouds_plot(pointcloud, pts) :
   
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')  # 1 row, 3 columns, subplot 1

   for ptc, c in zip([pointcloud, pts], ['r', 'b']) :
      x = ptc[..., 0].flatten()
      y = ptc[..., 1].flatten()
      z = ptc[..., 2].flatten()
      ax.scatter(x, y, z, s=10, c=c)

   ax.view_init(elev=90, azim=-90)  # Set elevation to 90 degrees to invert the z-axis

   plt.tight_layout()

   # Display the figure
   plt.show()


import pickle
from cloth_training.model.model_architecture.model_utils import depth_to_point_cloud, random_point_sampling, farthest_point_sampling, world_transform
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
   dataset_path = '/home/kgalassi/code/cloth/gym-cloth-main/logs/data-2023-08-04-14-27-seed-43-tier-3-seed-43-obs-blender-depth-False-rgbd-True-tier3_epis_10.pkl'
   with open(dataset_path, 'rb') as f:
      dataset = pickle.load(f)


   agent = HeatPerceiver(**model['init_params'])
   agent.heat_predictor.load_state_dict(model['heat_model'])
   agent.point_prediction.load_state_dict(model['point_model'])
   agent.action_predciton.load_state_dict(model['action_model'])
   agent.to('cuda')
   agent.eval()

   counter = 0

   print('Start experiment')
   with torch.no_grad():

      for i,ep in enumerate(dataset) :
         for k,step in enumerate(ep['act']) :
            act   = ep['act'][k]
            rgb   = torch.tensor(ep['obs'][k][:, :, :3])
            depth = torch.tensor(ep['obs'][k][:, :, 3])
            pts   = ep['pts'][k]
            pts   = torch.tensor(pts)


            # Pointcloud
            max = 1.4 + 0.002 #torch.min(pts[:,2])
            min = 1.4 - torch.max(pts[:,2])
            ptc = depth_to_point_cloud(depth, max=max, min=min)
            p = rgb.reshape(-1, 3)
            white_mask = (p[:, 0] > 200) & (p[:, 1] > 200) & (p[:, 2] > 200)
            ptc = ptc[~white_mask]  # Using ~ to negate the mask and keep non-white points
            ptc = random_point_sampling(ptc, 6000)     
            ptc = farthest_point_sampling(ptc, 1000)
            ptc = world_transform(ptc)
            ptc = torch.tensor(ptc).reshape(1,1000,3).to('cuda')

            poitnclouds_plot(ptc.reshape(-1,3).cpu().numpy(), pts.cpu().numpy())


            # Normalize the Gaussian tensor to have a maximum value of 1
            pick = torch.tensor(act[:2]) / 2.0 + 0.5
            distances2 = torch.norm(ptc[0,:,:2].cpu() - pick, dim=-1)
            min_distance_idx2 = torch.argmin(distances2)
            a_gt = act[2:]
            pt_gt = ptc[0,min_distance_idx2,:].squeeze(0).cpu().numpy()



            # forward + backward + optimize
            h,p,a = agent.forward(ptc)

            pt_taken = torch.argmax(p, dim=1)
            pt_taken = ptc[0,pt_taken,:].squeeze(0).cpu().numpy()

            a = a[0].squeeze(0).cpu().numpy()
            ptc = ptc.reshape(-1,3).cpu().numpy()


            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)


            pts = pts.cpu().numpy()
            ax1.scatter(ptc[:,0], ptc[:,1], c='b', s=3)
            ax1.scatter(pts[:,0], pts[:,1], c='m', s=3)
            ax1.scatter(pt_taken[0], pt_taken[1], c='r', s=10)
            ax1.arrow(x=pt_taken[0],y= pt_taken[1], dx=a[-2], dy=a[-1], width=0.008, color='red')
         
            ax1.scatter(pt_gt[0], pt_gt[1], c='green', s=20)
            ax1.arrow(x=pt_gt[0],y= pt_gt[1], dx=a_gt[-2], dy=a_gt[-1], width=0.008, color='green')


            ax2.imshow(rgb.cpu().numpy())

            plt.savefig(os.path.join(log_path,str(run_id), f'prediction{i}_{k}.png'))

            
         
