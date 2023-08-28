import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import os

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
   nn.init.orthogonal_(layer.weight, std)
   nn.init.constant_(layer.bias, bias_const)
   return layer



def set_seed(seed):
   if seed == None :
      print('Warning : seed is None')
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True


def update_print(*args):
   print(*args, end='\r', flush=True)


def get_precision_at_k(k, p, gt ) :
   # Get Precision@k
   topk_classes = torch.topk(p, k=k, dim=1)[1]  # Get the indices of top 5 predicted classes
   ground_truth_classes = torch.argmax(gt[:, :-2], dim=1)  # Get the ground truth classes

   correct_predictions = torch.zeros_like(ground_truth_classes).float()
   for i in range(k):
      correct_predictions += (topk_classes[:, i] == ground_truth_classes).float()

   return correct_predictions.sum().item() / len(gt)



def count_abs_error_below_threshold(distance_vector, threshold):
   errors = torch.abs(distance_vector)
   num_errors_below_threshold = torch.sum(errors < threshold)
   return num_errors_below_threshold.item()



def count_errors_in_range(distance_vector, lower_bound, upper_bound):
   errors = torch.logical_and(distance_vector >= lower_bound, distance_vector <= upper_bound)
   num_errors_in_range = torch.sum(errors)
   return num_errors_in_range.item()



def get_angle(vector1, vector2):
   dot_product = torch.sum(vector1 * vector2, dim=1)
   norms_product = torch.norm(vector1, dim=1) * torch.norm(vector2, dim=1)
   cosine_similarity = dot_product / norms_product
   angle = torch.acos(cosine_similarity)
   return angle


class DaggerDatasetTensor(Dataset):
   def __init__(self, obs, act):
      # build obserbation from list

      print('obs: ', type(obs))
      print('act: ', type(act))
      if type(obs) == list:
         self.observations = torch.stack(obs, dim=0)
         self.actions = torch.stack(act, dim=0)
      else :
         self.observations = obs
         self.actions = act

      print('obs: ', type(self.observations))
      print('act: ', type(self.actions))
      
   def __len__(self):
      #return len(self.observations)
      return self.observations.shape[0]

   def __getitem__(self, index):
      observation = self.observations[index, :]
      action = self.actions[index, :]

      return observation, action

   def save(self, path) :
      out = {
         'observations': self.observations,
         'actions': self.actions
      }
      
      with open(path, 'wb') as f:
         pickle.dump(out, f)
      print('saved completed')


class DaggerDataset(Dataset):
   def init(self, obs, act, shuffle_input=False):
      # build obserbation from list
      self.observations = torch.split(obs, 1, dim=0)
      self.actions = torch.split(act, 1, dim=0)
      
   def len(self):
      return len(self.observations)
   
   def getitem(self, index):
      observation = self.observations[index]
      action = self.actions[index]

      return observation, action

   def save_dataset(self, path):
      torch.save(self.observations, path + '/observations.pt')
      torch.save(self.actions, path + '/actions.pt')

   def load_dataset(self, path):
      self.observations = torch.load(path + '/observations.pt')
      self.actions = torch.load(path + '/actions.pt')




class DaggerDatasetSoftmax(Dataset):
   def __init__(self, obs=None, gt=None, path=None, shuffle_batch=False, gaussian_gt=False, seed =None):
      
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.shuffle_batch = shuffle_batch
      self.gaussion_gt = gaussian_gt
      set_seed(seed) if seed is not None else None
      
      if isinstance(path, str) :
         self.load_dataset(path)
         return
      

      if isinstance(obs, torch.Tensor) and isinstance(gt, torch.Tensor):
         # build obserbation from list
         self.observations = obs.clone().to(self.device)
         self.actions = gt.clone().to(self.device)

         if self.observations.shape[0] != self.actions.shape[0]:
            raise ValueError('Observation and action must have the same size')
         return 

      print('Empty Dataset created')


   def __len__(self):
      return self.observations.shape[0]
   
   def __getitem__(self, index):

      observation = self.observations[index, :]
      gt = self.actions[index, :]

      #gt[-2:] = torch.nn.functional.normalize(gt[-2:], dim=0) 

      if self.gaussion_gt :
         mu = gt.argmax()
         sigma = 0.5
         n = int(gt[:-2].shape[0] ** 0.5)
         xmu, ymu = mu % n, mu // n
         X, Y = torch.meshgrid(torch.linspace(0, n-1, n).to(self.device), torch.linspace(0, n-1, n).to(self.device), indexing='ij')
         Z = torch.exp(-((X-xmu)**2 + (Y-ymu)**2)/(2*sigma**2))

         gt[:-2] = Z.reshape(-1)

      if self.shuffle_batch :
         obs = observation.reshape(-1, 3)
         permuted_indices = torch.randperm(obs.size()[0], device=self.device)
         obs_like = obs[permuted_indices,:].reshape(-1)
         # permute the batch
         observation = obs_like
         gt[:-2] = gt[permuted_indices]
        
      return observation, gt

   def save_dataset(self, path):

      # build pickle
      data = {'observations': self.observations.cpu(), 
              'actions': self.actions.cpu()}

      # save pickle to path
      with open(path, 'wb') as f:
         pickle.dump(data, f)

   def load_dataset(self, path):

      # load pickle from path
      with open(path, 'rb') as f:
         data = pickle.load(f)

      #Read pickle
      self.observations = data['observations'].to(self.device)
      self.actions = data['actions'].to(self.device)




class DatasetImage(Dataset):

   def __init__(self, obs, gt, rgb, ptc, dtc, dtp, **kwargs) :

      self.obs = obs
      self.gt = gt
      self.rgb = rgb
      self.ptc = ptc
      self.dtc = dtc
      self.dtp = dtp

      self.permute = kwargs.get('permute', False)
      self.gaussian_gt = kwargs.get('gaussian_gt', False)

      assert len(self.obs) == len(self.gt) == len(self.rgb) == len(self.ptc) == len(self.dtc) == len(self.dtp)

      super().__init__()

   def extend_dataset(self, obs, gt, rgb, ptc, dtc, dtp) :
      self.obs.extend(obs)
      self.gt.extend(gt)
      self.rgb.extend(rgb)
      self.ptc.extend(ptc)
      self.dtc.extend(dtc)
      self.dtp.extend(dtp)

   def __len__(self):
      return len(self.obs)
   
   
   def __getitem__(self, index):

      obs = self.obs[index]
      gt = self.gt[index]
      rgb = self.rgb[index]
      ptc = self.ptc[index]
      dtc = self.dtc[index]
      dtp = self.dtp[index]

      if self.gaussian_gt :
         mu = gt.argmax()
         sigma = 0.5
         n = int(gt[:-2].shape[0] ** 0.5)

         xmu = mu % n
         ymu = mu // n

         x = torch.linspace(0, n-1, n).to(self.device)
         y = torch.linspace(0, n-1, n).to(self.device)
         X, Y = torch.meshgrid(x, y)
         Z = torch.exp(-((X-xmu)**2 + (Y-ymu)**2)/(2*sigma**2))

         gt[:-2] = Z.reshape(-1)
      
      if self.permute :
         obs = observation.reshape(-1, 3)
         permuted_indices = torch.randperm(obs.size()[0], device=self.device)
         obs_like = obs[permuted_indices,:].reshape(-1)
         observation = obs_like
         gt[:-2] = gt[permuted_indices]

      return obs, gt, rgb, ptc, dtc, dtp

   def set_gaussian_gt(self, value:bool) :
      self.gaussian_gt = value

   def set_permute(self, value:bool) :
      self.permute = value   



##############################################################################################################
# Plotting

import matplotlib.pyplot as plt

def create_offset_plot(obs,a,gt ):

   plt.figure(figsize=(10,10))
   obs_plot_pt = obs.reshape(-1, 3)[:,:2].cpu().numpy()
   plt.scatter(obs_plot_pt[:,0], obs_plot_pt[:,1], label='Observation')


   # Plot ground truth (max of gt)
   gt_idx = gt[:,:-2].argmax(dim=1)[0]
   plt.scatter(obs_plot_pt[gt_idx,0], obs_plot_pt[gt_idx,1], s=120, marker='x',color='green')


   # hard coded #TODO
   f_point = obs_plot_pt[gt_idx,:] + ((a*1.4) - 0.7).cpu().numpy()[0]
   f_gt    = obs_plot_pt[gt_idx,:] + ((gt[:, -2:]* 1.4) - 0.7).cpu().numpy()[0]


   plt.plot([obs_plot_pt[gt_idx,0], f_point[0]], 
            [obs_plot_pt[gt_idx,1], f_point[1]], color='red', linestyle='dashed', linewidth=2, label='Prediction')

   plt.plot([obs_plot_pt[gt_idx,0], f_gt[0]], 
            [obs_plot_pt[gt_idx,1], f_gt[1]], color='green', linestyle='dashed', linewidth=2, label='GT')
   
   plt.legend()
   return plt

 

def create_dual_offset_plot(obs,a,gt,base_path):
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

   # First plot: create_offset_plot
   obs_plot_pt = obs.reshape(-1, 3)[:,:2].cpu().numpy()
   ax1.scatter(obs_plot_pt[:,0], obs_plot_pt[:,1], label='Observation')

   gt_idx = gt[:,:-2].argmax(dim=1)[0]
   ax1.scatter(obs_plot_pt[gt_idx,0], obs_plot_pt[gt_idx,1], s=120, marker='x', color='green')

   f_point = obs_plot_pt[gt_idx,:] + ((a*1.4) - 0.7).cpu().numpy()[0]
   #f_point = obs_plot_pt[gt_idx,:] + ((gt[:, -2:]* 1.4) - 0.7).cpu().numpy()[0]
   f_gt    = obs_plot_pt[gt_idx,:] + ((gt[:, -2:]* 1.4) - 0.7).cpu().numpy()[0]

   ax1.plot([obs_plot_pt[gt_idx,0], f_point[0]],
            [obs_plot_pt[gt_idx,1], f_point[1]], color='red', linestyle='dashed', linewidth=2, label='Prediction')

   ax1.plot([obs_plot_pt[gt_idx,0], f_gt[0]],
            [obs_plot_pt[gt_idx,1], f_gt[1]], color='green', linestyle='dashed', linewidth=2, label='GT')

   ax1.set_title('Offset Plot')


   a = a.cpu().numpy()
   gt = gt.cpu().numpy()
   #gt_norm = gt_norm.cpu().numpy()

   ax2.plot([0.5, a[0,0]], [0.5, a[0,1]], color='red', linestyle='dashed', linewidth=2, label='Prediction')
   ax2.plot([0.5, gt[0, -2]], [0.5, gt[0, -1]], color='green', linestyle='dashed', linewidth=2, label='GT')
   #ax2.plot([0.5, gt_norm[0, -2]], [0.5, gt_norm[0, -1]], color='blue', linestyle='dashed', linewidth=2, label='GT Norm')

   ax2.set_title('Action Plot')
   ax2.set_xlim([0, 1.1])
   ax2.set_ylim([0, 1.1])
   ax2.set_xticks(np.arange(0, 1.0, 0.25))
   ax2.set_yticks(np.arange(0, 1.0, 0.25))

   plt.tight_layout()
   plt.legend()
   plt.savefig(base_path)
   plt.close()
   plt.clf()
   return plt



def create_dual_offset_plot_with_prediction(obs,a,p, gt,base_path):
   fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
   
   obs_plot_pt = obs.reshape(-1, 3)[:,:2].cpu().numpy()

   # PLOT CLOTH POINTS
   ax1.scatter(obs_plot_pt[:,0], obs_plot_pt[:,1], label='Observation')

   p_idx  = p[:,:-2].argmax(dim=1)[0]
   gt_idx = gt[:,:-2].argmax(dim=1)[0]

   # Plot GT displacement
   ax1.scatter(obs_plot_pt[gt_idx,0], obs_plot_pt[gt_idx,1], s=120, marker='x', color='green')


   # Plot line between P, action predicted   
   f_point = obs_plot_pt[p_idx,:] + ((a*1.4) - 0.7).cpu().numpy()[0]
   ax1.plot([obs_plot_pt[p_idx,0], f_point[0]],
            [obs_plot_pt[p_idx,1], f_point[1]], color='red', linestyle='dashed', linewidth=2, label='Prediction')

   # Plot of GT point and GT displacement
   f_gt    = obs_plot_pt[gt_idx,:] + ((gt[:, -2:]* 1.4) - 0.7).cpu().numpy()[0]
   ax1.plot([obs_plot_pt[gt_idx,0], f_gt[0]],
            [obs_plot_pt[gt_idx,1], f_gt[1]], color='green', linestyle='dashed', linewidth=2, label='GT')

   ax1.set_title('Offset Plot')


   a = a.cpu().numpy()
   gt = gt.cpu().numpy()
   #gt_norm = gt_norm.cpu().numpy()
   '''
   ax2.plot([0.5, a[0,0]], [0.5, a[0,1]], color='red', linestyle='dashed', linewidth=2, label='Prediction')
   ax2.plot([0.5, gt[0, -2]], [0.5, gt[0, -1]], color='green', linestyle='dashed', linewidth=2, label='GT')
   #ax2.plot([0.5, gt_norm[0, -2]], [0.5, gt_norm[0, -1]], color='blue', linestyle='dashed', linewidth=2, label='GT Norm')

   ax2.set_title('Action Plot')
   ax2.set_xlim([0, 1.1])
   ax2.set_ylim([0, 1.1])
   ax2.set_xticks(np.arange(0, 1.0, 0.25))
   ax2.set_yticks(np.arange(0, 1.0, 0.25))
   '''
   plt.tight_layout()
   plt.legend()
   plt.savefig(base_path)
   plt.close()
   plt.clf()
   return plt


def create_prob_plot(obs,a,gt ) :

   plt.figure(figsize=(10,10))
   obs_plot_pt = obs.reshape(-1, 3)[:,:2].cpu().numpy()
   plt.scatter(obs_plot_pt[:,0], obs_plot_pt[:,1], label='Observation')

   # select where pt is > 0.7 and where is max

   p = F.softmax(p, dim=1)
   fifth_highest_value = torch.sort(p, descending=True).values[0,4]

   idx = torch.where(p > fifth_highest_value)[1]
   max_id = torch.argmax(p, dim=1)


   # Plot ground truth (max of gt)
   gt_idx = gt[:,:-2].argmax(dim=1)[0]
   plt.scatter(obs_plot_pt[gt_idx,0], obs_plot_pt[gt_idx,1], label='GT', s=120, marker='x',color='green')
   
   prob = p[:,idx].cpu().numpy()

   for j, id in enumerate(idx):
      # Plot all prediction with scatter
      plt.scatter(obs_plot_pt[id,0], obs_plot_pt[id,1], label=f'Prediction {id}', color='red')
      
      label = prob[0,j]
      pt = obs_plot_pt[id,:]
      if id == max_id:
         plt.annotate(f'{label:.2f}', (pt[0], pt[1]), textcoords="offset points", xytext=(0, 10), ha='center', weight='bold', fontsize=12)
      else:
         plt.annotate(f'{label:.2f}', (pt[0], pt[1]), textcoords="offset points", xytext=(0, 10), ha='center',fontsize=12)

   plt.legend()
   return plt




def create_combined_plot(obs, a, gt, p, base_path):
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

   # First plot: create_offset_plot
   obs_plot_pt = obs.reshape(-1, 3)[:,:2].cpu().numpy()
   ax1.scatter(obs_plot_pt[:,0], obs_plot_pt[:,1], label='Observation')

   gt_idx = gt[:,:-2].argmax(dim=1)[0]
   ax1.scatter(obs_plot_pt[gt_idx,0], obs_plot_pt[gt_idx,1], s=120, marker='x', color='green')

   f_point = obs_plot_pt[gt_idx,:] + ((a*1.4) - 0.7).cpu().numpy()[0]
   f_gt = obs_plot_pt[gt_idx,:] + ((gt[:, -2:]* 1.4) - 0.7).cpu().numpy()[0]

   ax1.plot([obs_plot_pt[gt_idx,0], f_point[0]],
            [obs_plot_pt[gt_idx,1], f_point[1]], color='red', linestyle='dashed', linewidth=2, label='Prediction')

   ax1.plot([obs_plot_pt[gt_idx,0], f_gt[0]],
            [obs_plot_pt[gt_idx,1], f_gt[1]], color='green', linestyle='dashed', linewidth=2, label='GT')

   ax1.set_title('Offset Plot')




   # Second plot: create_prob_plot
   ax2.scatter(obs_plot_pt[:,0], obs_plot_pt[:,1], label='Observation')


   p = F.softmax(p, dim=1)
   fifth_highest_value = torch.sort(p, descending=True).values[0,4]
   idx = torch.where(p > fifth_highest_value)[1]
   max_id = torch.argmax(p, dim=1)

   gt_idx = gt[:,:-2].argmax(dim=1)[0]
   ax2.scatter(obs_plot_pt[gt_idx,0], obs_plot_pt[gt_idx,1], label='GT', s=120, marker='x', color='green')

   prob = p[:,idx].cpu().numpy()

   for j, id in enumerate(idx):
      ax2.scatter(obs_plot_pt[id,0], obs_plot_pt[id,1], label=f'Prediction {id}', color='red')

      label = prob[0,j]
      pt = obs_plot_pt[id,:]
      if id == max_id:
         ax2.annotate(f'{label:.2f}', (pt[0], pt[1]), textcoords="offset points", xytext=(0, 10), ha='center', weight='bold', fontsize=12)
      else:
         ax2.annotate(f'{label:.2f}', (pt[0], pt[1]), textcoords="offset points", xytext=(0, 10), ha='center',fontsize=12)

   ax2.set_title('Probability Plot')

   plt.tight_layout()
   plt.legend()
   plt.savefig(base_path)
   plt.close()
   plt.clf()
   return plt


def create_histogram(errors, bin_limits, path):
   # Append negative infinity to the beginning of bin limits
   #bin_limits.insert(0, -np.inf)

   # Append positive infinity to the end of bin limits
   #bin_limits.append(np.inf)

   # Create the histogram
   plt.clf()


   hist, bins = np.histogram(errors, bins=bin_limits)

   # Plot the histogram
   plt.bar(bins[:-1], hist, width=np.diff(bins))

   plt.xlabel('Error')
   plt.ylabel('Occurrences')
   

   xticks = (bin_limits[1:] - bin_limits[:-1])/2 + bin_limits[:-1]
   plt.xticks(xticks, rotation = 45)

   
   plt.title('Error Histogram')
   plt.savefig(path)

   # Convert the plot to a tensor
   #fig = plt.gcf()
   #fig.canvas.draw()
   #image_tensor = np.array(fig.canvas.renderer.buffer_rgba())

   # Clear the plot

   #return plt


"""Lamb optimizer."""

import collections
import math

import torch
from torch.optim import Optimizer

'''
from tensorboardX import SummaryWriter
def log_lamb_rs(optimizer: Optimizer, event_writer: SummaryWriter, token_count: int):
   """Log a histogram of trust ratio scalars in across layers."""
   results = collections.defaultdict(list)
   for group in optimizer.param_groups:
      for p in group['params']:
         state = optimizer.state[p]
         for i in ('weight_norm', 'adam_norm', 'trust_ratio'):
               if i in state:
                  results[i].append(state[i])

   for k, v in results.items():
      event_writer.add_histogram(f'lamb/{k}', torch.tensor(v), token_count)

'''

class Lamb(Optimizer):
   r"""Implements Lamb algorithm.

   It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

   Arguments:
      params (iterable): iterable of parameters to optimize or dicts defining
         parameter groups
      lr (float, optional): learning rate (default: 1e-3)
      betas (Tuple[float, float], optional): coefficients used for computing
         running averages of gradient and its square (default: (0.9, 0.999))
      eps (float, optional): term added to the denominator to improve
         numerical stability (default: 1e-8)
      weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
      adam (bool, optional): always use trust ratio = 1, which turns this into
         Adam. Useful for comparison purposes.

   .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
      https://arxiv.org/abs/1904.00962
   """

   def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
               weight_decay=0, adam=False):
      if not 0.0 <= lr:
         raise ValueError("Invalid learning rate: {}".format(lr))
      if not 0.0 <= eps:
         raise ValueError("Invalid epsilon value: {}".format(eps))
      if not 0.0 <= betas[0] < 1.0:
         raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
      if not 0.0 <= betas[1] < 1.0:
         raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
      defaults = dict(lr=lr, betas=betas, eps=eps,
                     weight_decay=weight_decay)
      self.adam = adam
      super(Lamb, self).__init__(params, defaults)

   def step(self, closure=None):
      """Performs a single optimization step.

      Arguments:
         closure (callable, optional): A closure that reevaluates the model
               and returns the loss.
      """
      loss = None
      if closure is not None:
         loss = closure()

      for group in self.param_groups:
         for p in group['params']:
            if p.grad is None:
               continue
            grad = p.grad.data
            if grad.is_sparse:
               raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

            state = self.state[p]

            # State initialization
            if len(state) == 0:
               state['step'] = 0
               # Exponential moving average of gradient values
               state['exp_avg'] = torch.zeros_like(p.data)
               # Exponential moving average of squared gradient values
               state['exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            state['step'] += 1

            # Decay the first and second moment running average coefficient
            # m_t
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # v_t
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Paper v3 does not use debiasing.
            # bias_correction1 = 1 - beta1 ** state['step']
            # bias_correction2 = 1 - beta2 ** state['step']
            # Apply bias to lr to avoid broadcast.
            step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

            weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

            adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
            if group['weight_decay'] != 0:
               adam_step.add_(p.data, alpha=group['weight_decay'])

            adam_norm = adam_step.pow(2).sum().sqrt()
            if weight_norm == 0 or adam_norm == 0:
               trust_ratio = 1
            else:
               trust_ratio = weight_norm / adam_norm
            state['weight_norm'] = weight_norm
            state['adam_norm'] = adam_norm
            state['trust_ratio'] = trust_ratio
            if self.adam:
               trust_ratio = 1

            p.data.add_(adam_step, alpha=-step_size * trust_ratio)

      return loss
   



import itertools, pickle,os, re
def iterate_hyperparameters(hyperparameters):
   keys = list(hyperparameters.keys())
   values = [hyperparameters[key] if isinstance(hyperparameters[key], list) else [hyperparameters[key]] for key in keys]
   combinations = list(itertools.product(*values))

   for combination in combinations:
      yield dict(zip(keys, combination)) 

def iterate_hyperparameters_with_exclusion(hyperparameters):
   keys = list(hyperparameters.keys())
   values = [hyperparameters[key] if isinstance(hyperparameters[key], list) else [hyperparameters[key]] for key in keys]
   combinations = list(itertools.product(*values))

   c_remove = []
   for combination in combinations:
      hparams = dict(zip(keys, combination))
      offset_network_type = hparams['offset_network_type']

      if offset_network_type in [0, 1]:
         hparams.pop('offset_hidden_dim', None)
         hparams.pop('inj_hidden_dim', None)

      elif offset_network_type == 2:
         hparams.pop('inj_hidden_dim', None)
      c_remove.append(hparams)

   unique_combinations = []
      
   for combination in c_remove:
      if combination not in unique_combinations:
         unique_combinations.append(combination)
         
   for combination in unique_combinations:
      yield combination
      #yield dict(zip(keys, combination)) # combination is already the dictionary

def load_dictionary(dataset_path, pattern = r"ag_dataset(\d+)\.pickle", no_dataset=-1):
   '''
   Load dictionary at the given path
   @param dataset_path: path to the dataset
   @param pattern: pattern to match the dataset file name
   @param no_dataset: index of the dataset to load. -1 to load the last dataset

   Return pickle dataset
   '''
   
   # Find all the .pickle files in the folder matching the naming pattern
   folder_path = os.path.join(dataset_path, '*.pickle')
   # Get a list of files in the directory
   file_list = os.listdir(dataset_path)
   # Extract the dataset number value from the file names
   file_tuples = []
   for file_name in file_list:
      match = re.search(pattern, file_name)
      if match:
         dataset_number = int(match.group(1))
         file_tuples.append((file_name, dataset_number))

   # Sort the file list based on the extracted integer value
   sorted_files = sorted(file_tuples, key=lambda x: x[1])
   print(f'Obtained {len(sorted_files)} files: ')

   if no_dataset == -1:
      file_name, dataset_number = sorted_files[-1]
   else:
      for loaded_file_name, loaded_dataset_number in sorted_files:
         if loaded_dataset_number == no_dataset:
            file_name = loaded_file_name
            dataset_number = loaded_dataset_number
            break

   file_path = os.path.join(dataset_path, file_name)
   print('Loading dataset ', file_path)
   with open(file_path, 'rb') as f:
      data = pickle.load(f)  

   return data

def get_run_counter(log_dir):
   
   run_counter = 0
   # Iterate over the existing log directories

   if not os.path.exists(log_dir):
      os.makedirs(log_dir)
      return 0

   while os.path.exists(os.path.join(log_dir, str(run_counter))):
      run_counter += 1
      
   return run_counter + 1


class EarlyStopper:
    def __init__(self, patience=5, mode='min'):
        self.patience = patience
        self.mode = mode
        self.best_metric = np.inf if mode == 'min' else -np.inf
        self.no_improvement_count = 0
        self.stopped_epoch = 0
        self.early_stop = False
        
    def should_stop(self, metric):
        if self.mode == 'min':
            improvement = metric < self.best_metric
        else:
            improvement = metric > self.best_metric
        
        if improvement:
            self.no_improvement_count = 0
            self.best_metric = metric
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.early_stop = True
        return self.early_stop
    

  


#################################################
########## POINTCLOUD OPERATION   ###############
#################################################
RATIO = 1
FOCAL_LENGTH_MM, SENSOR_WIDTH_MM = 40.0, 36
IMAGE_WITDH_PIXELS,IMAGE_HEIGHT_PIXELS = 224, 224
CAMERA_TRANSFORM = [ 1.0, 0.0,  0.0, 0.5,
                     0.0, -1.0, 0.0, 0.5,
                     0.0, 0.0,  -1.0, 1.4,
                     0.0, 0.0,  0.0, 1.0]
FOCAL_LENGTH_PIXEL_X = FOCAL_LENGTH_MM / SENSOR_WIDTH_MM * IMAGE_WITDH_PIXELS
FOCAL_LENGTH_PIXEL_Y = FOCAL_LENGTH_MM / SENSOR_WIDTH_MM * IMAGE_HEIGHT_PIXELS
CX, CY = IMAGE_WITDH_PIXELS / 2.0, IMAGE_HEIGHT_PIXELS / 2.0
INTRINSIC_PARAMS = [FOCAL_LENGTH_PIXEL_X, 0, CX,
                    0, FOCAL_LENGTH_PIXEL_Y, CY,
                    0, 0, 1]


def random_point_sampling(pointcloud, n) :
   sampled_indices = [torch.randint(0, len(pointcloud), (n,))]
   sampled_points = pointcloud[sampled_indices].reshape(n,3)
   return sampled_points

def random_point_sampling_tensor(pointcloud, n) :
   b, num_points, _ = pointcloud.shape

   # Generate random indices for each point cloud in the batch
   sampled_indices = torch.randint(0, num_points, size=(b, n))

   # Create an index tensor for advanced indexing
   index_tensor = torch.arange(0, b).unsqueeze(1).expand(-1, n)

   # Use advanced indexing to select the random points from each point cloud
   sampled_points = pointcloud[index_tensor, sampled_indices]
   return sampled_points

def farthest_point_sampling(point_cloud, num_samples):
   """
   Farthest Point Sampling algorithm for point clouds.

   Args:
      point_cloud (torch.Tensor): Point cloud tensor of shape (N, 3) where N is the number of points.
      num_samples (int): Number of points to sample.

   Returns:
      torch.Tensor: Indices of the sampled points.
   """
   
   point_cloud = point_cloud.to(torch.device('cuda'))

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


   return point_cloud[sampled_indices].to(torch.device('cpu'))


def depth_to_point_cloud(depth, max, min, intrinsic_params=INTRINSIC_PARAMS):
   # get image dimensions
   height, width = depth.shape
   depth_max = torch.max(depth).float()
   depth = min + depth/depth_max * (max - min)
   
   # unpack intrinsic parameters
   fx, _, cx, _, fy, cy, _, _, _ = intrinsic_params
   
   # create meshgrid of pixel coordinates
   x, y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='ij')
   
   # compute 3D coordinates of each pixel
   x3d = (x - cx) * depth / fx / RATIO
   y3d = (y - cy) * depth / fy / RATIO
   z3d = depth / RATIO 
   
   point_cloud = torch.stack([x3d, y3d, z3d], axis=-1)
   point_cloud = point_cloud.reshape(-1, 3)

   return point_cloud

def depth_to_point_cloud_tensor(depth, max, min, intrinsic_params=INTRINSIC_PARAMS):
   #depth Nx224x224x1
   #Max Nx1
   #Min Nx1

   n, height, width = depth.shape
   #depth_max = torch.max(depth, dim=(1, 2))[0].unsqueeze(1)
   max_values, _ = torch.max(depth, dim=2)

   # Step 2: Reshape the tensor to Nx1
   depth_max, _ = torch.max(max_values, dim=1, keepdim=True)

   depth = (min + (depth.reshape(n,-1)/depth_max) * (max - min)).reshape(n, width, height)
   fx, _, cx, _, fy, cy, _, _, _ = intrinsic_params
   # compute 3D coordinates of each pixel
   x, y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='ij')
   x3d = (x - cx) * depth / fx / RATIO
   y3d = (y - cy) * depth / fy / RATIO
   z3d = depth / RATIO 
   
   point_cloud = torch.stack([x3d, y3d, z3d], axis=-1)
   point_cloud = point_cloud.reshape(n, -1, 3)

   return point_cloud


def world_transform(pointcloud):
   ones = torch.ones((pointcloud.shape[0], 1))
   ptc_h = torch.hstack([pointcloud, ones])
   t = torch.tensor(CAMERA_TRANSFORM).reshape(4,4)

   ptc_out = torch.empty_like(pointcloud)
   for i in range(ptc_h.shape[0]):
      ptc_out[i] = torch.matmul(t, ptc_h[i])[:3]
   return ptc_out

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

