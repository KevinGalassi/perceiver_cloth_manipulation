# Attention-Based Cloth Manipulation from Model-free Topological Representation

2024 IEEE International Conference on Robotics and Automation (ICRA)

### Abstract
The robotic manipulation of deformable objects, such as clothes and fabric, is known as a complex task from both the perception and planning perspectives. Indeed, the stochastic nature of the underlying environment dynamics makes it an interesting research field for statistical learning approaches and neural policies. In this work, we introduce a novel attention-based neural architecture capable of solving a smoothing task for such objects by means of a single robotic arm. To train our network, we leverage an oracle policy, executed in simulation, which uses the topological description of a mesh of points for representing the object to smooth. In a second step, we transfer the resulting behavior in the real world with imitation learning using the cloth point cloud as decision support, which is captured from a single RGBD camera placed egocentrically on the wrist of the arm. This approach allows fast training of the real-world manipulation neural policy while not requiring scene reconstruction at test time, but solely a point cloud acquired from a single RGBD camera. Our resulting policy first predicts the desired point to choose from the given point cloud and then the correct displacement to achieve a smoothed cloth. Experimentally, we first assess our results in a simulation environment by comparing them with an existing heuristic policy, as well as several baseline attention architectures. Then, we validate the performance of our approach in a real-world scenario.

## Content of the repository

This repository contains the pytorch implementation of the perceiver-nased network used to learn a cloth-smoothing policy using imitation learning.\\

The synthetic dataset used for the training dataset is collected using a mass-spring-damper simulator derived from https://github.com/DanielTakeshi/gym-cloth

## Citation
If you find the code or other related resources useful, please consider citing the paper:

```
@inproceedings{10610241,
  author={Galassi, Kevin and Wu, Bingbing and Perez, Julien and Palli, Gianluca and Renders, Jean-Michel},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Attention-Based Cloth Manipulation from Model-free Topological Representation}, 
  year={2024},
  pages={18207-18213},

```
