
# Reacher

This is Pytorch implementation for reinforcement learning project `Reacher`

## Introduction

In this environment, a double-jointed arm can move to target locations.  

A reward of +0.1 is provided for each step that the agent's hand is in the goal location.   

The goal of your agent is to maintain its position at the target location for as many time steps as possible.

## Prerequisites
* Python 3.6
* Pytorch 1.4

## Installation
* Clone this repo:

```
git clone https://github.com/BaldwinHe/ProfolioOfBai
cd PortfolioOfBai/Reinforcement Learning/Reacher/
```
* Install python requirements:

```
pip install -r requirements.txt
```
* Install unity environment file based on your operating system
    
## Getting Started

### 0. Quick Testing

To see how our trained model performs in collecting bananas, run the code below:

```
python test.py --unity [path to unity environment] --checkpoint ./checkpoints/DQN/checkpointFC_final.pth
```

### 1. Training


```
python train.py --unity [path to unity environment] 
```

### 2. Testing

```
python test.py --unity [path to unity environment] --checkpoint [path to checkpoint]
```
### 3. Apply DQN improvement methods


## Results

### 0.Comparison between inexperienced and trained agent


### 1. Plot of rewards per episode


## Ideas for Future Work

