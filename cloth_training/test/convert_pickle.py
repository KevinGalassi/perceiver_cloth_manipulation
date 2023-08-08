import pickle
import torch

if __name__ == '__main__': 

   prob_path = '/home/kgalassi/omniverse/omniverse_estension_deformable/exts/franka.cloth/franka/cloth/dagger_network/network_model/separated_network/offset_network/06-12-16-30.pickle'
   pth_path  = '/home/kgalassi/omniverse/omniverse_estension_deformable/exts/franka.cloth/franka/cloth/dagger_network/network_model/separated_network/offset_network/06-12-16-30.pth'

   with open(prob_path, 'rb') as f:
      data = pickle.load(f)

   print(data.keys())    
   torch.save(data, pth_path)

   print('saved 1')
   exit()


   pickle_path = '/home/kgalassi/omniverse/omniverse_estension_deformable/exts/franka.cloth/franka/cloth/dagger_network/network_model/cascade_network/offset_network/06-07-17-59.pickle'
   pth_path    = '/home/kgalassi/omniverse/omniverse_estension_deformable/exts/franka.cloth/franka/cloth/dagger_network/network_model/cascade_network/offset_network/06-07-17-59.pth'


   with open(prob_path, 'rb') as f:
      data = pickle.load(f)

   print(data.keys())    
   torch.save(data, pth_path)
