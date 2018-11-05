'''
This code implements a simple heuristic for the CartPole_V0 environment
'''

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np



##########################################
###############BEGIN PROGRAM##############
##########################################

env = gym.make('CartPole-v0')

print(env.action_space)

n_episodes = 1
maxStates = 10**4  # function of bin discritization
gamma = 0.9        # discount factor
alpha = 0.01       # step size

scores = []
for i_episode in range(n_episodes):
    env.reset()
    for t in range(300):
        env.render()

        if t == 0:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

        else:
            if observation[3] > 0:
                action = 1
            else:
                action = 0
            observation, reward, done, info = env.step(action)
        print("\nState: " + str(observation))
        print("Reward: " + str(reward))
        print("Done: " + str(done))

        #print(env.step(action))
        if done:
            print("\nEpisode  finished  at  t = " + str(t+1))
            break
    env.close()
    scores.append(t+1)

print("\nAverage Score: " + str(float(sum(scores))/float(len(scores))))
