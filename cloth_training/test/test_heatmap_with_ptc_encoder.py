



import torch
from cloth_training.model.model_architecture.dataset_gen import GymClothDataset
from cloth_training.model.perceiver_heatmap import HeatPerceiver
from cloth_training.model.pointcloud_encoder import PointCloudEncoder

import os, pickle
from tqdm import tqdm
from torch.utils.data import DataLoader


from cloth_training.model.model_architecture.model_utils import get_precision_at_k


if __name__ == '__main__':

   folder_name = 'heat6'
   write_log = True
   save_model = True

   run_id = '07-27-13-23'
   run_id = '07-27-02-50'
   #run_id = '07-27-07-39'

   model_path = './model/saved_model/heat_map'
   log_path = './model/logs/heat_map_ptc_encoder'
   pkl_path = '/home/kgalassi/code/cloth/gym-cloth-simulator/logs'
   pkl_name = 'evaluation-transformer-07-25-22-37-3-seed-None-obs-blender-depth-False-rgbd-True-tier3_epis_10'


   # PTC MODEL
   ptc_run_id = '07-28-17-42'
   ptc_model_path = './model/saved_model/ptc_encoder'


   model = torch.load(os.path.join(model_path, run_id + '.pth'))
   hparams = model['init_params']


   from torch.utils.tensorboard import SummaryWriter
   print('Logging to ', os.path.join(log_path, str(run_id)))
   writer = SummaryWriter(log_dir=os.path.join(log_path, str(run_id)))
   writer.add_text('Hyperparameters', str(hparams))
   writer.flush()




   with open(os.path.join(pkl_path, pkl_name + '.pkl'), 'rb') as f:
      loaded_pkl = pickle.load(f)

   episode_list = [episode for episode in loaded_pkl] 
   action_list, obs_list, pts_list = [], [], []

   print(f'Found {len(episode_list)} episodes')

   for episode in episode_list:
      for i in range(len(episode['act'])) :
         action_list.append(torch.tensor(episode['act'][i]))
         obs_list.append(torch.tensor(episode['obs'][i]))
         pts_list.append(torch.tensor(episode['pts'][i]))


   dataset = GymClothDataset(action_list, obs_list, pts_list)
   #torch.save(episode_list, os.path.join(pkl_path + '.pt'))

   dataset.set_obs_type('heatmap+ptc')
   dataset.set_output_type('heatmap+ptc')
   dataset.to_device(torch.device('cpu'))
   # Not shuffle points
   test_loader  = DataLoader(dataset, batch_size=1, shuffle=True)
   print('Dataset and dataloader ok')

   distance, distance_x, distance_y, angle = 0.0, 0.0, 0.0, 0.0
   distance_d, distance_dx, distance_dy = 0.0, 0.0, 0.0

   test_results = []   


   agent = HeatPerceiver(**model['init_params'])
   agent.heat_predictor.load_state_dict(model['heat_model'])
   agent.point_prediction.load_state_dict(model['point_model'])
   agent.action_predciton.load_state_dict(model['action_model'])
   agent.to('cuda')
   agent.eval()


   ptc_model = torch.load(os.path.join(ptc_model_path, ptc_run_id + '.pth'))
   ptc_encoder = PointCloudEncoder(**ptc_model['init_params'])
   ptc_encoder.load_state_dict(ptc_model['model'])
   ptc_encoder.to('cuda')
   ptc_encoder.eval()


   counter = 0
   with torch.no_grad():
      for i,data in tqdm(enumerate(test_loader)) :
         obs, gt = data


         # Get batch size and number of points
         b = obs[0].shape[0]
         n_pts = obs[0].shape[1]
         permuted_indices = torch.randperm(n_pts)
         #permuted_indices = torch.arange(n_pts)

         # Convert dataset to tensor in cuda  
         pts = obs[0][:, permuted_indices].to('cuda')
         ptc = obs[2].to('cuda')
         
         # get GT
         gt_gaussian     = gt[0][:,permuted_indices].to('cuda')
         gaussian_tensor = gt[0]                      
         gt_action       = gt[1].to('cuda')
         action_gt = gt_action[:,2:].reshape(-1,2)

         # Get prediction with ptc encoder
         pts_e = ptc_encoder.forward(ptc)
         ptc_h, ptc_p, ptc_a = agent.forward(pts_e)


         pt_idx = gt_gaussian.argmax(dim=-1)
         gt_prob = torch.zeros_like(gt_gaussian).to('cuda')
         gt_prob[torch.where(gt_gaussian == 1)] = 1

         #Get Logs
         acc = (torch.argmax(ptc_p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
         precision_at_4 = get_precision_at_k(5, ptc_p, gt_prob)
         precision_at_3 = get_precision_at_k(3, ptc_p, gt_prob)
         precision_at_2 = get_precision_at_k(2, ptc_p, gt_prob)

         pt_taken, pt_gt = torch.argmax(ptc_p, dim=1), torch.argmax(gt_prob, dim=1)
         pt_pos, gt_pos =  pts_e[torch.arange(pts_e.shape[0]), pt_taken], pts_e[torch.arange(pts_e.shape[0]), pt_gt]
         distance_prob = torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b

         distance_d = torch.sqrt(torch.sum((ptc_a.squeeze(0) - gt_action[:,2:])**2, dim=1)).sum().item() / b
         distance_dx = torch.sqrt(torch.sum((ptc_a[:,:,-2] - gt_action[:,-2])**2, dim=0)).sum().item() / b
         distance_dy = torch.sqrt(torch.sum((ptc_a[:,:,-1] - gt_action[:,-1])**2, dim=0)).sum().item() / b
         angle = torch.sqrt(torch.sum((torch.atan2(ptc_a[:,:,-1], ptc_a[:,:,-2]) - torch.atan2(gt_action[:,-1], gt_action[:,-2]))**2, dim=0)).sum().item() / b


         # Save Images
         print(ptc_h.shape)
         print(torch.max(ptc_h))
         print(torch.min(ptc_h))
         heatmap = agent.get_matplotlib_heatmap(ptc_h, gaussian_tensor, permuted_indices)
         heatmap.savefig(os.path.join(log_path,str(run_id), f'heatmap{i}.png'))
         
         point = agent.get_matplotlib_action2(pts, pts_e, pt_taken, pt_gt,ptc_a, action_gt)
         point.savefig(os.path.join(log_path,str(run_id), f'prediction{i}.png'))
         


         '''
         # Get Prediction with grounf truth
         h,p,a = agent.forward(pts)

         pt_idx = gt_gaussian.argmax(dim=-1)

         gt_prob = torch.zeros_like(gt_gaussian).to('cuda')
         gt_prob[torch.where(gt_gaussian == 1)] = 1

         #Get accuracy
         acc = (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
         precision_at_4 = get_precision_at_k(5, p, gt_prob)
         precision_at_3 = get_precision_at_k(3, p, gt_prob)
         precision_at_2 = get_precision_at_k(2, p, gt_prob)

         pt_taken, pt_gt = torch.argmax(p, dim=1), torch.argmax(gt_prob, dim=1)
         pt_pos, gt_pos =  pts[torch.arange(pts.shape[0]), pt_taken], pts[torch.arange(pts.shape[0]), pt_gt]
         distance_prob = torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b

         distance_d = torch.sqrt(torch.sum((a.squeeze(0) - gt_action[:,2:])**2, dim=1)).sum().item() / b
         distance_dx = torch.sqrt(torch.sum((a[:,:,-2] - gt_action[:,-2])**2, dim=0)).sum().item() / b
         distance_dy = torch.sqrt(torch.sum((a[:,:,-1] - gt_action[:,-1])**2, dim=0)).sum().item() / b
         angle = torch.sqrt(torch.sum((torch.atan2(a[:,:,-1], a[:,:,-2]) - torch.atan2(gt_action[:,-1], gt_action[:,-2]))**2, dim=0)).sum().item() / b

         
         '''
         
         val_step = {
                     'accuracy' : acc,
                     'precision4': precision_at_4,
                     'precision3': precision_at_3,
                     'precision2': precision_at_2,
                     'distance_prob': distance_prob,
                     'distance_d': distance_d,
                     'distance_dx': distance_dx,
                     'distance_dy': distance_dy,
                     'angle': angle
                     }
         test_results.append(val_step)

         counter += 1

         heatmap = agent.get_matplotlib_heatmap(ptc_h, gaussian_tensor, permuted_indices)
         heatmap.savefig(os.path.join(log_path,str(run_id), f'heatmap{i}.png'))

         action_gt = gt_action[:,2:].reshape(-1,2)

         point = agent.get_matplotlib_action(pts, pt_taken, pt_gt,ptc_a, action_gt)
         point.savefig(os.path.join(log_path,str(run_id), f'prediction{i}.png'))


      
   print('Results :')
   print(f'Accuracy : {sum([result["accuracy"] for result in test_results])/counter}')
   print(f'distance : {sum([result["distance_d"] for result in test_results])/counter}')
   print(f'distance_dx : {sum([result["distance_dx"] for result in test_results])/counter}')
   print(f'distance_dy : {sum([result["distance_dy"] for result in test_results])/counter}')
   print(f'angle : {sum([result["angle"] for result in test_results])/counter}')

   if write_log :
      for num, result in enumerate(test_results) :
         for key, value in result.items():
            writer.add_scalar(f'Test/{key}', value, num)

         writer.flush()