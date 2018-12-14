'''
This code tests a policy in the CartPole_V0 environment


format for weight vector update and implementation from:
https://github.com/kvfrans/openai-cartpole

'''

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import random
import timeit


def run_episode(env, parameters):
    observation = env.reset()
    cumulative_reward = 0
    max_steps = 200
    for step in range(max_steps):
        if np.matmul(parameters, observation) < 0:
            action = 0
        else:
            action = 2
        #print(env.step(action))
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            return cumulative_reward

##########################################
###############BEGIN PROGRAM##############
##########################################
env = gym.make('MountainCar-v0')    #The mountaincar source code can be edited to perform different tests
#env._max_episode_steps = 500

visualization = False
exportFilename = "Hill Climbing MountainCar.txt"
trials = 50
iters = 150

start_time = timeit.default_timer()

###SIMULATE###
noise_scaling = 0.2
parameters = np.random.rand(2) * 2 - 1
parameters = [-1.87949625e-04,  5.72014565e-01]  #New
oldR = []
for _ in range(trials):
    oldR.append(run_episode(env, parameters))
bestreward = np.mean(oldR)
print("Best Reward: " + str(bestreward))
print("Best Parameters: " + str(parameters))
for _ in range(iters):
    newparams = parameters + (np.random.rand(2) * 2 - 1)*noise_scaling
    #newparams = parameters
    reward = []
    for i in range(trials):
        reward.append(run_episode(env, newparams))
    #print(np.mean(reward))
    #print(np.mean(reward))
    if np.mean(reward) > bestreward:
        bestreward = np.mean(reward)
        parameters = newparams
        print("\nCurrent Best: " + str(bestreward))
        print("Best Parameters: " + str(parameters))
        if np.mean(reward) == 0:
            break
print("\nBest Reward: " + str(bestreward))
print(parameters)

elapsed = timeit.default_timer() - start_time
print("\nTIME: " + str(elapsed))
