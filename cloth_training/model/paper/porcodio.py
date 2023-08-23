import torch

p = torch.randn([3,10,1])
print(p)

t = torch.zeros_like(p)
print(max_indices.shape)
print(max_indices)

# Now you can set the corresponding values in gt_prob to 1
# Set the corresponding values in gt_prob to 1
max_indices = torch.argmax(p, dim=1)
t.scatter_(1, max_indices.unsqueeze(1), 1)
print(t)