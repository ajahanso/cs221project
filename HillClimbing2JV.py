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


def assign_bins(observation, bins):
    state = [0]*4
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    return state


def get_state_as_string(state):
    string_state = ''.join(str(int(e)) for e in state)
    return string_state


def run_episode(env, parameters):
    observation = env.reset()
    cumulative_reward = 0
    max_steps = 500
    for step in range(max_steps):
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            return cumulative_reward

##########################################
###############BEGIN PROGRAM##############
##########################################
env = gym.make('CartPole-v0')    #The cartpole source code can be edited to perform different tests
env._max_episode_steps = 500

visualization = False
stateActionPairs = {}
exportFilename = "Hill Climbing.txt"
maxStates = 10**4
trials = 50

all_states = []
for i in range(maxStates):
    all_states.append(str(i).zfill(4))



###CREATE DISCRETE BINS###
#These should match the bins used for the policy determination!
bins = np.zeros((4, 10))              # 10 discretizations for each variable
bins[0] = np.linspace(-4.8, 4.8, 10)  # observation[0]: cart position: -4.8 - 4.8
bins[1] = np.linspace(-5, 5, 10)      # observation[1]: cart velocity: -inf - inf
bins[2] = np.linspace(-.4, .4, 10)    # observation[2]: pole angle: -41.8 - 41.8
bins[3] = np.linspace(-5, 5, 10)      # observation[3]: pole velocity: -inf - inf


start_time = timeit.default_timer()

###SIMULATE###
noise_scaling = 0.1
parameters = np.random.rand(4) * 2 - 1
bestreward = 0
for _ in range(10000):
    newparams = parameters + (np.random.rand(4) * 2 - 1)*noise_scaling
    reward = []
    for i in range(trials):
        reward.append(run_episode(env, newparams))
    print(np.mean(reward))
    if np.mean(reward) > bestreward:
        bestreward = np.mean(reward)
        parameters = newparams
        if np.mean(reward) == 500:
            break
print("Best Reward: " + str(bestreward))
print(parameters)

#OUTPUT LEARNED POLICY TO FILE###
file = open(exportFilename, "w")
file.write("s,a") #CSV file: state,action
for state in all_states:
    observation = [4.8 * (-1 + 0.2 * int(state[0]) - 0.1), 5 * (-1 + 0.2 * int(state[1]) - 0.1),
                   0.4 * (-1 + 0.2 * int(state[2]) - 0.1), 5 * (-1 + 0.2 * int(state[3]) - 0.1)]
    act = 0 if np.matmul(parameters, observation) < 0 else 1
    file.write("\n" + str(state) + "," + str(act)) #CSV file: state,action
file.close()


elapsed = timeit.default_timer() - start_time
print("TIME: " + str(elapsed))
