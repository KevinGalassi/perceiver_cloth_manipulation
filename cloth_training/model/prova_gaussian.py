import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class FixedCovMixture:
    """ The model to estimate gaussian mixture with fixed covariance matrix. """
    def __init__(self, n_components, cov, max_iter=100, random_state=None, tol=1e-10):
        self.n_components = n_components
        self.cov = cov
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol=tol

    def fit(self, X):
        # initialize the process:
        np.random.seed(self.random_state)
        n_obs, n_features = X.shape
        self.mean_ = X[np.random.choice(n_obs, size=self.n_components)]
        # make EM loop until convergence
        i = 0
        for i in range(self.max_iter):
            new_centers = self.updated_centers(X)
            if np.sum(np.abs(new_centers-self.mean_)) < self.tol:
                break
            else:
                self.mean_ = new_centers
        self.n_iter_ = i

    def updated_centers(self, X):
        """ A single iteration """
        # E-step: estimate probability of each cluster given cluster centers
        cluster_posterior = self.predict_proba(X)
        # M-step: update cluster centers as weighted average of observations
        weights = (cluster_posterior.T / cluster_posterior.sum(axis=1)).T
        new_centers = np.dot(weights, X)
        return new_centers


    def predict_proba(self, X):
        likelihood = np.stack([multivariate_normal.pdf(X, mean=center, cov=self.cov) 
                               for center in self.mean_])
        cluster_posterior = (likelihood / likelihood.sum(axis=0))
        return cluster_posterior

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=0)
    
n = 25
gt = torch.zeros((n,n))
def draw_circle(center, variance, ax=None, **kwargs):
    """Draw a circle with a given center and variance"""
    ax = ax or plt.gca()
    radius = np.sqrt(variance)
    circle = plt.Circle(center, radius, **kwargs)
    ax.add_patch(circle)

gt[2,2] = 1
gt[2,23] = 1
gt[23,2] = 1
gt[23,23] = 1

std = 2

# Generate coordinate grids
x = torch.arange(n).float()
y = torch.arange(n).float()
grid_x, grid_y = torch.meshgrid(x, y)

# Get the coordinates of the element equal to 1
coord_xs, coord_ys = torch.where(gt == 1)

actions_gaussian = torch.zeros((n,n))

for coord_x, coord_y in zip(coord_xs, coord_ys):
   coord_x = coord_x.unsqueeze(0)
   coord_y = coord_y.unsqueeze(0)
   # Calculate the Gaussian distribution centered at the specified coordinates
   mean = torch.stack([coord_x.float(), coord_y.float()], dim=-1)
   std = std  # Standard deviation of the Gaussian distribution
   gaussian_tensor = torch.exp(-((grid_x - mean[:, 0])**2 + (grid_y - mean[:, 1])**2) / (2 * std**2))

   # Normalize the Gaussian tensor to have a maximum value of 1
   actions_gaussian += (gaussian_tensor / torch.max(gaussian_tensor))


dataset = actions_gaussian.flatten()

mask = dataset > 0.01
# Reshape 'grid_x' and 'grid_y' to match the dataset shape
grid_x_reshaped = grid_x.flatten()[mask].numpy()
grid_y_reshaped = grid_y.flatten()[mask].numpy()

# Stack 'grid_x_reshaped' and 'grid_y_reshaped' to get the input features for GMM
X = np.column_stack((grid_x_reshaped, grid_y_reshaped))

# Fit Gaussian Mixture Model
num_components = 4  # You can change this to the desired number of Gaussian components
inv_cov = np.ones(num_components)* (1/std)

gmm = GaussianMixture(n_components=num_components, 
                           #covariance_type='spherical', 
                           init_params='kmeans',
                           random_state=42,
                           max_iter=300,
                           #precisions_init=inv_cov,
                           tol=1e-6)

gmm.fit(X)

# Get the means and spherical variances of the Gaussian components
means = gmm.means_
spherical_variances = gmm.covariances_

# Print the means and spherical variances of the Gaussian components
print("Gaussian Means:")
print(means)
print("Spherical Variances:")
print(spherical_variances)

# Visualize the Gaussian Mixture Model
plt.scatter(X[:, 0], X[:, 1], s=5, alpha=0.5)
for i in range(num_components):
    draw_circle(means[i], spherical_variances[i], alpha=0.5, color='red')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Circular Gaussian Mixture Model')
plt.show()
