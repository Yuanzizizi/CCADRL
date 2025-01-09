# Cooperative Motion Planning in Divided Environments via Congestion-Aware Deep Reinforcement Learning

This repository contains the **official implementation** of the paper:  
**"Cooperative Motion Planning in Divided Environments via Congestion-Aware Deep Reinforcement Learning"**  
Authors: Yuanyuan Du, Jianan Zhang, Xiang Cheng, and Shuguang Cui  
Published in *IEEE Robotics and Automation Letters (RA-L)*, December 2024.

## Abstract

This work proposes a novel cooperative motion planning algorithm leveraging **Congestion-Aware Deep Reinforcement Learning (CCADRL)** to address collisions and congestion in environments divided by narrow hallways. Key contributions include:
1. A temporal arrival intent sharing paradigm that is used
 for constructing a hallway map, informing asynchronous
 individual motion planning around hallways.
 2) A non-myopic congestion-aware scheme that incorpo
rates a hallway goal chooser and a congestion predictor.
 This scheme prevents the agent from adhering to heavily
 congested trajectories that may be only slightly shorter
 and enables the agent to decide whether to claim getting
 into or avoid the selected hallway.
 3) A relation analyzer that encodes interaction dynam
ics among neighboring agents, enriching the agents’
 decision-making capabilities.

Simulations demonstrate significant improvements over state-of-the-art algorithms in various challenging scenarios.

<center> <img src="https://i-blog.csdnimg.cn/direct/8af988ab6d114b9f90143c626b02981c.png" width="100%"></center>

## Repository Status

- This repository includes:
  - Evaluation scripts.
  - Simulated environments.
    
    The [gym environment code](https://github.com/mit-acl/gym-collision-avoidance) is included as a submodule.

  - Pre-trained models.

    It can be found in path: `gym-collision-avoidance\gym_collision_avoidance\experiments\src\checkpoints`

## How to Use

- Installation steps.
  Grab the code from github, initialize submodules, install dependencies and src code

  ```bash
  # Clone either through SSH or HTTPS
  git clone --recursive git@github.com:Yuanzizizi/CCADRL.git

  ```
- Examples for testing CCADRL.
  ```bash
  ./CCADRL_demo.sh
  ```
  - results can be found in `gym-collision-avoidance\gym_collision_avoidance\experiments\results`

## Citation

If you find this repository helpful, please consider citing our paper:
@ARTICLE{10829688, 
  author={Du, Yuanyuan and Zhang, Jianan and Cheng, Xiang and Cui, Shuguang},
  journal={IEEE Robotics and Automation Letters}, 
  title={Cooperative Motion Planning in Divided Environments via Congestion-Aware Deep Reinforcement Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={Planning;Uncertainty;Navigation;Deep reinforcement learning;Collision avoidance;Cognition;Observability;Decision making;Analytical models;Trajectory;Motion planning;collision avoidance;congestion-aware;deep reinforcement learning},
  doi={10.1109/LRA.2025.3526448}}




