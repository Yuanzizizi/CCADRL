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

## Repository Status

- This repository includes:
  - Evaluation scripts.
  - Simulated environments.
  - Pre-trained models and detailed documentation.

## How to Use

- Installation steps.
- Examples for testing CCADRL.

## Citation

If you find this repository helpful, please consider citing our paper:



