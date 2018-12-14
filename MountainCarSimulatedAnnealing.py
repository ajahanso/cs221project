'''
This code tests a policy in the CartPole_V0 environment
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
iters = 1000

start_time = timeit.default_timer()


###SIMULATE###
startTemp = 0.5
parameters = np.random.rand(2) * 2 - 1
bestreward = -float("Inf")
for k in range(iters):
    currentTemp = startTemp - 0.99*startTemp*(k/iters)
    newparams = parameters + (np.random.rand(2) * 2 - 1)*currentTemp
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
