import torch

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

def filter_background(rgb:torch.Tensor, ptc:torch.Tensor):
   p = rgb.reshape(-1, 3)
   assert p.shape == ptc.shape, 'Point cloud and RGB image have different shapes'
   white_mask = (p[:, 0] > 200) & (p[:, 1] > 200) & (p[:, 2] > 200)
   ptc = ptc[~white_mask]  # Using ~ to negate the mask and keep non-white points
   print('White mask : ', ptc.shape,' loaded')

   backgroud_idx = white_mask.nonzero(as_tuple=True)[0]
   cloth_idx = (~white_mask).nonzero(as_tuple=True)[0]

   return ptc, backgroud_idx, cloth_idx






def world_transform(ptc) :
   ones = torch.ones((ptc.shape[0], 1))
   ptc_h = torch.hstack([ptc, ones])
   t = torch.tensor(CAMERA_TRANSFORM).reshape(4,4)

   ptc2 = torch.empty((ptc_h.shape[0], 3))
   for i in range(ptc_h.shape[0]):
      outm = torch.matmul(t, ptc_h[i])
      ptc2[i] = outm[:3]
   return ptc2



def rdn_index(ptc, n) :
   return torch.randint(0, len(ptc), (n,))



def fps_index(indices, point_cloud, num_samples):
   """
   Farthest Point Sampling algorithm for point clouds.

   Args:
      point_cloud (torch.Tensor): Point cloud tensor of shape (N, 3) where N is the number of points.
      num_samples (int): Number of points to sample.

   Returns:
      torch.Tensor: Indices of the sampled points.
   """
   
   num_points = indices.shape[0]
   sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=point_cloud.device)
   distances = torch.full((num_points,), float('inf'), device=point_cloud.device)
   farthest_idx = torch.randint(0, num_points, (1,), dtype=torch.long, device=point_cloud.device)
   for i in range(num_samples):
      sampled_indices[i] = farthest_idx
      farthest_point = point_cloud[farthest_idx]
      dist_to_farthest = torch.norm(point_cloud[indices] - farthest_point, dim=1)
      mask = dist_to_farthest < distances  
      distances[mask] = dist_to_farthest[mask]
      farthest_idx = indices[torch.argmax(distances)]


   return sampled_indices



def get_background_idx(rgb:torch.Tensor, ptc:torch.Tensor, idx):
   p = rgb.reshape(-1, 3)
   white_mask = (p[:, 0] > 200) & (p[:, 1] > 200) & (p[:, 2] > 200)

   white_mask = white_mask[idx]

   #ptc = ptc[~white_mask]  # Using ~ to negate the mask and keep non-white points
   #print('White mask : ', ptc.shape,' loaded')

   backgroud_idx = white_mask.nonzero(as_tuple=True)[0]
   cloth_idx = (~white_mask).nonzero(as_tuple=True)[0]

   return backgroud_idx, cloth_idx
