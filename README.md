# Phasic Policy Gradient

Simple code to demonstrate Deep Reinforcement Learning by using Phasic Policy Gradient in Pytorch & Tensorflow.
There are some modification on this code, which may have many differences than original implementation

The code is use newer version of PPO called Truly PPO (instead of PPO Clip), which has more sample efficiency and performance than PPO Clip. 

- [x] Use Pytorch and Tensorflow 2
- [x] Use Truly PPO
- [ ] Add more complex environment
- [ ] Add more explanation

## Getting Started

This project is using Pytorch and Tensorflow 2 for Deep Learning Framework, Gym for Reinforcement Learning Environment.
Although it's not required, but i recommend run this project on a PC with GPU and 8 GB Ram

### Prerequisites

Make sure you have installed Pytorch and Gym.  
- Click [here](https://gym.openai.com/docs/) to install gym

You can use either Pytorch or Tensorflow 2
- Click [here](https://pytorch.org/get-started/locally/) to install pytorch
- Click [here](https://www.tensorflow.org/install) to install tensorflow 2

### Installing

Just clone this project into your work folder

```
git clone https://github.com/wisnunugroho21/phasic_policy_gradient.git
```

## Running the project

After you clone the project, run following script in cmd/terminal :

#### Pytorch version
```
cd phasic_policy_gradient/discrete/pytorch/
python ppg_dis.py
```

#### Tensorflow 2 version
```
cd phasic_policy_gradient/discrete/tensorflow
python ppg_dis_tf.py
```

## Proximal Policy Optimization

PPO is motivated by the same question as TRPO: how can we take the biggest possible improvement step on a policy using the data we currently have, without stepping so far that we accidentally cause performance collapse? Where TRPO tries to solve this problem with a complex second-order method, PPO is a family of first-order methods that use a few other tricks to keep new policies close to old. PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

There are two primary variants of PPO: PPO-Penalty and PPO-Clip.

* PPO-Penalty approximately solves a KL-constrained update like TRPO, but penalizes the KL-divergence in the objective function instead of making it a hard constraint, and automatically adjusts the penalty coefficient over the course of training so that it’s scaled appropriately.

* PPO-Clip doesn’t have a KL-divergence term in the objective and doesn’t have a constraint at all. Instead relies on specialized clipping in the objective function to remove incentives for the new policy to get far from the old policy.

OpenAI use PPO-Clip  
You can read full detail of PPO in [here](https://arxiv.org/abs/1707.06347)

## Truly Proximal Policy Optimization

Proximal policy optimization (PPO) is one of the most successful deep reinforcement-learning methods, achieving state-of-the-art performance across a wide range of challenging tasks. However, its optimization behavior is still far from being fully understood. In this paper, we show that PPO could neither strictly restrict the likelihood ratio as it attempts to do nor enforce a well-defined trust region constraint, which means that it may still suffer from the risk of performance instability. To address this issue, we present an enhanced PPO method, named Truly PPO. Two critical improvements are made in our method: 1) it adopts a new clipping function to support a rollback behavior to restrict the difference between the new policy and the old one; 2) the triggering condition for clipping is replaced with a trust region-based one, such that optimizing the resulted surrogate objective function provides guaranteed monotonic improvement of the ultimate policy performance. It seems, by adhering more truly to making the algorithm proximal - confining the policy within the trust region, the new algorithm improves the original PPO on both sample efficiency and performance.

You can read full detail of Truly PPO in [here](https://arxiv.org/abs/1903.07940)

## Phasic Policy Gradient

We introduce Phasic Policy Gradient (PPG), a reinforcement learning framework which modifies traditional on-policy actor-critic methods by separating policy and value function training into distinct phases. In prior methods, one must choose between using a shared network or separate networks to represent the policy and value function. Using separate networks avoids interference between objectives, while using a shared network allows useful features to be shared. PPG is able to achieve the best of both worlds by splitting optimization into two phases, one that advances training and one that distills features. PPG also enables the value function to be more aggressively optimized with a higher level of sample reuse. Compared to PPO, we find that PPG significantly improves sample efficiency on the challenging Procgen Benchmark.

You can read full detail of Phasic Policy Gradient in [here](https://arxiv.org/abs/2009.04416)

## Result

### LunarLander

| Result Gif  |
| ------------- |
| ![Result Gif](https://github.com/wisnunugroho21/asynchronous_PPO/blob/master/Result/lunarlander.gif)  |

### Bipedal

| Result Gif    |
| ------------- |
| ![Result Gif](https://github.com/wisnunugroho21/asynchronous_PPO/blob/master/Result/bipedal.gif) |

### Pendulum

| Result Gif  |
| ------------- |
| ![Result Gif](https://github.com/wisnunugroho21/asynchronous_PPO/blob/master/Result/pendulum.gif)  | 

## Contributing
This project is far from finish and will be improved anytime . Any fix, contribute, or idea would be very appreciated
