'''
This code tests a policy in the CartPole_V0 environment
'''

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd


def assign_bins(observation, bins):
    state = [0]*4
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    return state


def get_state_as_string(state):
    string_state = ''.join(str(int(e)) for e in state)
    return string_state



##########################################
###############BEGIN PROGRAM##############
##########################################
env = gym.make('CartPole-v0') #The cartpole source code can be edited to perform different tests
n_episodes = 20
visualization = False
scores = []
stateActionPairs = {}

df = pd.read_csv("Policy Simple Q.txt", dtype=str)
#Column list to access dataframe later
columns = list(df)

for i in range(len(df)):
    #Form possible states
    stateActionPairs[df.at[i, 's']] = df.at[i, 'a']


###CREATE DISCRETE BINS###
#These should match the bins used for the policy determination!
bins = np.zeros((4, 10))              # 10 discretizations for each variable
bins[0] = np.linspace(-4.8, 4.8, 10)  # observation[0]: cart position: -4.8 - 4.8
bins[1] = np.linspace(-5, 5, 10)      # observation[1]: cart velocity: -inf - inf
bins[2] = np.linspace(-.4, .4, 10)    # observation[2]: pole angle: -41.8 - 41.8
bins[3] = np.linspace(-5, 5, 10)      # observation[3]: pole velocity: -inf - inf



###SIMULATE###
for i_episode in range(n_episodes):
    for t in range(300):
        if t == 0:
            observation = env.reset()  # initialize to semi-random start state

        if visualization:
            env.render() #Graphics on or off

        state = get_state_as_string(assign_bins(observation, bins))
        action = int(stateActionPairs[state])
        observation, reward, done, info = env.step(action)

        if done:
            print("\nEpisode  finished  at  t = " + str(t+1))
            break
    env.close()
    scores.append(t+1)

print("\nAverage Score: " + str(float(sum(scores))/float(len(scores))))
