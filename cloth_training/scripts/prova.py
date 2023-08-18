import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




import torch


from cloth_training.model.common.attention_models import Attention, FeedForward


from scipy.spatial.distance import cdist

def farthest_point_sampling(point_cloud, n):
   sampled_points = np.zeros((n, point_cloud.shape[1]))
   sampled_indices = [np.random.randint(0, len(point_cloud))]
   for i in range(1, n):
      distances = cdist(point_cloud, point_cloud[sampled_indices])
      min_distances = np.min(distances, axis=1)
      new_index = np.argmax(min_distances)
      sampled_indices.append(new_index)
   sampled_points = point_cloud[sampled_indices]
   return sampled_points



n=1000
t = torch.tensor(np.random.rand(1544, 3))
p = torch.tensor(farthest_point_sampling(t, n))

print(p.shape)
exit()

cross = Attention(embed_dim=625, num_heads=1, batch_first=True,dropout=0.0)

input = torch.randn(1, 10, 625)
context = torch.randn(1, 625, 625)


idx = torch.randperm(10)
idx2 = torch.randperm(625)

out1 = cross(input, context)
out2 = cross(input[:, idx, :], context)

print('Diff data, same context')
print(out1 - out2)


print('Same data, diff context')
out3 = cross(input, context[:, idx2, :])
print(out1 - out3)

print('Diff data, diff context')
out4 = cross(input[:, idx, :], context[:, idx2, :])
print(out1 - out4)


print('1) ', torch.max(out1 - out2))
print('2) ', torch.max(out1 - out3))
print('3) ', torch.max(out1 - out4))
exit()
# Replace these arrays with your actual 3D images
image1 = np.random.rand(10, 10, 10)
image2 = np.random.rand(10, 10, 10)
image3 = np.random.rand(10, 10, 10)

# Create a new figure
fig = plt.figure(figsize=(30, 10))

# Create three subplots within the figure
ax1 = fig.add_subplot(131, projection='3d')  # 1 row, 3 columns, subplot 1
ax2 = fig.add_subplot(132, projection='3d')  # 1 row, 3 columns, subplot 2
ax3 = fig.add_subplot(133, projection='3d')  # 1 row, 3 columns, subplot 3

# Plot the 3D images in the respective subplots
ax1.voxels(image1, edgecolor='k')
ax2.voxels(image2, edgecolor='k')
ax3.voxels(image3, edgecolor='k')

# Add titles to the subplots (optional)
ax1.set_title('Image 1')
ax2.set_title('Image 2')
ax3.set_title('Image 3')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Display the figure
plt.show()