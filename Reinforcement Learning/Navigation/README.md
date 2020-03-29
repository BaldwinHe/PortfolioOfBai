
# Banana Collector

This is Pytorch implementation for reinforcement learning project `Banana Collector`

## Introduction

An intelligent agent use DQN to navigate and collect bananas! in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  

The goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

## Prerequisites
* Python 3.6
* Pytorch 1.4

## Installation
* Clone this repo:

```
git clone https://github.com/BaldwinHe/ProfolioOfBai
cd PortfolioOfBai/Reinforcement Learning/Navigation/
```
* Install python requirements:

```
pip install -r requirements.txt
```
* Install unity environment file:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)
    
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
* `--double` : Double DQN

## Results

### 0.Comparison between inexperienced and trained agent
![banana collector demo](https://github.com/BaldwinHe/DemoLibrary/blob/master/Reinforcement%20Learning/Banana%20Collector/banana_result.gif)

### 1. Plot of rewards per episode
![score](https://github.com/BaldwinHe/DemoLibrary/blob/master/Reinforcement%20Learning/Banana%20Collector/score_result.png)

## Ideas for Future Work
> try some DQN improvment methods

* [x] Double DQN(DDQN)
* [ ] Prioritized Experience Replay(Prioritized DQN)
* [ ] Dueling DQN
* [ ] A3C
* [ ] Distributional DQN
* [ ] Noisy DQN
* [ ] Rainbow