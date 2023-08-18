from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import torch

def compute_surface_curvature(point_cloud, k_neighbors=20):
   """
   Compute the curvature of a point cloud using principal curvatures.

   Args:
      point_cloud (torch.Tensor): Nx3 tensor representing the 3D point cloud.
      k_neighbors (int, optional): Number of neighbors to consider when computing curvature. Default is 20.

   Returns:
      torch.Tensor: Nx2 tensor containing the principal curvatures for each point.
   """
   n_points = point_cloud.shape[0]

   # Initialize tensors to store the principal curvatures
   curvatures     = torch.zeros((n_points, 2))
   curvature_norm = torch.zeros(n_points)
   normals        = torch.zeros((n_points, 3))

   for i in range(n_points):
      # Calculate the squared Euclidean distance from the current point to all points in the cloud
      distances = torch.sum((point_cloud - point_cloud[i])**2, dim=1)
      
      # Get the indices of the k-nearest neighbors
      _, indices = torch.topk(distances, k_neighbors, largest=False)

      # Extract the neighborhood points
      neighborhood = point_cloud[indices]

      # Compute the local covariance matrix
      centroid = torch.mean(neighborhood, dim=0)
      centered = neighborhood - centroid
      covariance_matrix = torch.matmul(centered.t(), centered) / k_neighbors

      # Compute the eigenvalues of the covariance matrix
      eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)


      # Principal curvatures are the eigenvalues in ascending order
      curvatures[i] = eigenvalues[:2]

      curvature = eigenvalues.min() / eigenvalues.sum()
      curvature_norm[i] = curvature

      curvature = eigenvalues.min() / eigenvalues.sum()

      # Get the eigenvector corresponding to the smallest eigenvalue, which represents the normal of the plane
      normal = eigenvectors[:, 0]
      # Ensure the normal points outward (towards the center of the neighborhood)
      if torch.dot(normal, centroid - point_cloud[i]) > 0:
         normal *= -1
      normals[i] = normal

   return curvatures, curvature_norm, normals


def compute_heat_kernel_signatures(points, sigma, num_points, num_steps, device='cpu'):
   """
   Compute heat kernel signatures for a point cloud using PyTorch.

   Arguments:
      points (torch.Tensor): Input point cloud (N, 3).
      sigma (float): Heat kernel parameter.
      num_points (int): Number of points to be sampled on the point cloud.
      num_steps (int): Number of diffusion steps to compute the heat kernel.
      device (str): Device to run the computation on, e.g., 'cpu', 'cuda'.

   Returns:
      torch.Tensor: Heat kernel signatures for the input point cloud (N, num_points).
   """
   # Convert points to torch.Tensor and move to the desired device
   points = torch.tensor(points, dtype=torch.float32).to(device)

   # Randomly sample num_points from the input point cloud
   selected_indices = torch.randperm(points.shape[0])[:num_points]
   selected_points = points[selected_indices]

   # Compute pairwise distances between points
   pairwise_distances = torch.cdist(selected_points, selected_points)

   # Compute heat kernel matrix
   heat_kernel_matrix = torch.exp(-pairwise_distances ** 2 / (2 * sigma ** 2))

   # Normalize the heat kernel matrix
   heat_kernel_matrix = heat_kernel_matrix / heat_kernel_matrix.sum(dim=1, keepdim=True)

   # Compute the heat kernel signatures using matrix multiplication
   heat_kernel_signatures = heat_kernel_matrix
   for _ in range(num_steps - 1):
      heat_kernel_signatures = torch.matmul(heat_kernel_signatures, heat_kernel_matrix)

   return heat_kernel_signatures

def compute_fpfh_1(point_cloud, radius, num_neighbors):
   # Initialize the FPFH feature array
   fpfh_features = torch.zeros((point_cloud.shape[0], num_neighbors), dtype=np.float32)

   # Create a KDTree for fast nearest neighbor search
   kdtree = cKDTree(point_cloud)

   # Find neighbors for each point
   _, neighbors = kdtree.query(point_cloud, k=num_neighbors+1)

   def compute_fpfh_for_point(query_point, neighbors, radius):
      # Compute the distance between the query point and its neighbors
      distances = torch.linalg.norm(neighbors - query_point, axis=1)

      # Compute the FPFH feature by counting unique neighbors within a certain radius
      unique_counts = torch.unique(neighbors, axis=0, return_counts=True)[1]

      # Normalize the counts by the volume of the shell defined by the radius
      fpfh_feature = unique_counts / (torch.pi * radius ** 2 * distances.sum())

      return fpfh_feature

   for i, (point, neighbor_indices) in enumerate(zip(point_cloud, neighbors)):
      # Remove the first entry as it corresponds to the point itself
      neighbor_indices = neighbor_indices[1:]

      # Compute FPFH for the current point
      fpfh = compute_fpfh_for_point(point, point_cloud[neighbor_indices], radius)

      # Store the FPFH feature in the feature array
      fpfh_features[i] = torch.tensor(fpfh)

   return fpfh_features

