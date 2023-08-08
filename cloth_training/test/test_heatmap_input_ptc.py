



import torch
from cloth_training.model.model_architecture.dataset_gen import GymClothDataset
from cloth_training.model.perceiver_heatmap import HeatPerceiver

import os
from tqdm import tqdm
from torch.utils.data import DataLoader


from cloth_training.model.model_architecture.model_utils import iterate_hyperparameters, set_seed
from cloth_training.model.model_architecture.model_utils import get_precision_at_k

if __name__ == '__main__' :
   write_log = True
   save_model = True

   run_id = '07-27-13-23'
   run_id = '07-27-02-50'
   #run_id = '07-27-07-39'

   model_path = './model/saved_model/heat_map'
   log_path = './model/logs/heat_map_input_ptc'


   model = torch.load(os.path.join(model_path, run_id + '.pth'))
   hparams = model['init_params']


   from torch.utils.tensorboard import SummaryWriter
   print('Logging to ', os.path.join(log_path, str(run_id)))
   writer = SummaryWriter(log_dir=os.path.join(log_path, str(run_id)))
   writer.add_text('Hyperparameters', str(hparams))
   writer.flush()



   dataset = torch.load('/home/kgalassi/code/cloth/cloth_training/cloth_training/dataset/ptc/dataset.pt')
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

   counter = 0
   print('Start experiment')
   with torch.no_grad():
      for i,data in tqdm(enumerate(test_loader), total=len(test_loader)) :
         obs, gt = data


         # Get batch size and number of points
         b = obs[0].shape[0]
         n_pts = obs[0].shape[1]
         permuted_indices = torch.randperm(n_pts)
         #permuted_indices = torch.arange(n_pts)

         # Convert dataset to tensor in cuda  
         pts = obs[0][:, permuted_indices].to('cuda')
         gt_gaussian = gt[0][:,permuted_indices].to('cuda')
         gaussian_tensor = gt[0]                      
         gt_action = gt[1].to('cuda')
         ptc = obs[2].to('cuda')
         pts += torch.tensor([0.5, 0.5, 0.0], device=ptc.device)


         # forward + backward + optimize
         h,p,a = agent.forward(ptc)

         pt_idx = gt_gaussian.argmax(dim=-1)

         gt_prob = torch.zeros_like(gt_gaussian).to('cuda')
         gt_prob[torch.where(gt_gaussian == 1)] = 1

         #Get accuracy
         acc = (torch.argmax(p, dim=1) == torch.argmax(gt_prob, dim=1)).sum().item() / b
         precision_at_4 = get_precision_at_k(5, p, gt_prob)
         precision_at_3 = get_precision_at_k(3, p, gt_prob)
         precision_at_2 = get_precision_at_k(2, p, gt_prob)

         pt_taken, pt_gt = torch.argmax(p, dim=1), torch.argmax(gt_prob, dim=1)
         pt_pos, gt_pos =  ptc[torch.arange(b), pt_taken], ptc[torch.arange(b), pt_gt]
         distance_prob = torch.sqrt(torch.sum((pt_pos - gt_pos)**2, dim=1)).sum().item() / b

         distance_d = torch.sqrt(torch.sum((a.squeeze(0) - gt_action[:,2:])**2, dim=1)).sum().item() / b
         distance_dx = torch.sqrt(torch.sum((a[:,:,-2] - gt_action[:,-2])**2, dim=0)).sum().item() / b
         distance_dy = torch.sqrt(torch.sum((a[:,:,-1] - gt_action[:,-1])**2, dim=0)).sum().item() / b
         angle = torch.sqrt(torch.sum((torch.atan2(a[:,:,-1], a[:,:,-2]) - torch.atan2(gt_action[:,-1], gt_action[:,-2]))**2, dim=0)).sum().item() / b

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

         #heatmap = agent.get_matplotlib_heatmap(h, gaussian_tensor, permuted_indices)
         #heatmap.savefig(os.path.join(log_path,str(run_id), f'heatmap{i}.png'))

         #heatmap = agent.get_matplotlib_heatmap2(ptc, h, gaussian_tensor)
         #heatmap.savefig(os.path.join(log_path,str(run_id), f'heatmap{i}.png'))


         action_gt = gt_action[:,2:].reshape(-1,2)

         point = agent.get_matplotlib_action2(pts, ptc, pt_taken, pt_gt,a, action_gt)
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