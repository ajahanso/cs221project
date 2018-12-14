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
env._max_episode_steps = 500

trials = 20
n_episodes = 20
discrete = 10
visualization = False
scores = []
stateActionPairs = {}
inputFilename = "Policy Simple Q keep.txt"
#inputFilename = "Hill Climbing keep.txt"
#inputFilename = "Heuristic Policy keep.txt"
#inputFilename = "Simulated Annealing.txt"


#inputFilename = "Policy Simple Q fine.txt"
#discrete = 20

Hill = False
weightVecHill = [0.00420598, 0.04623107, 0.74504948, 0.53752899]
#weightVecHill = [0.04986472, 0.05068857, 0.73239703, 0.70470094]  #Trained on tau=0.03

Annealing = False
weightVecAnneal = [-0.22652625, 1.33992206, 0.82230289, 2.04780219]
weightVecAnneal = [0.06507162, 1.48161965, 1.11192081, 1.37986328]  #Trained on tau=0.03

HillMountain = False
weightHillMountain = [-1.87949625e-04,  5.72014565e-01]  #Standard
weightHillMountain = [-0.02747171,  0.94514477]  #Trained on doubled mass I think

if HillMountain:
    env = gym.make('MountainCar-v0') #The cartpole source code can be edited to perform different tests
    env._max_episode_steps = 200

AnnealMountain = False
weightAnnealMountain = [-0.00150503,  1.40005323]
if AnnealMountain:
    env = gym.make('MountainCar-v0') #The cartpole source code can be edited to perform different tests
    env._max_episode_steps = 200



df = pd.read_csv(inputFilename, dtype=str)
#Column list to access dataframe later
columns = list(df)

for i in range(len(df)):
    #Form possible states
    stateActionPairs[df.at[i, 's']] = df.at[i, 'a']


###CREATE DISCRETE BINS###
#These should match the bins used for the policy determination!
bins = np.zeros((4, discrete))              # 10 discretizations for each variable
bins[0] = np.linspace(-4.8, 4.8, discrete)  # observation[0]: cart position: -4.8 - 4.8
bins[1] = np.linspace(-5, 5, discrete)      # observation[1]: cart velocity: -inf - inf
bins[2] = np.linspace(-.4, .4, discrete)    # observation[2]: pole angle: -41.8 - 41.8
bins[3] = np.linspace(-5, 5, discrete)      # observation[3]: pole velocity: -inf - inf

###SIMULATE###
averages = []
for i_trial in range(trials):
    for i_episode in range(n_episodes):
        for t in range(env._max_episode_steps):
            if t == 0:
                observation = env.reset()  # initialize to semi-random start state

            if visualization:
                env.render() #Graphics on or off

            if Hill:
                action = 0 if np.matmul(weightVecHill, observation) < 0 else 1
            elif Annealing:
                action = 0 if np.matmul(weightVecAnneal, observation) < 0 else 1
            elif HillMountain:
                    action = 0 if np.matmul(weightHillMountain, observation) < 0 else 2
            elif AnnealMountain:
                action = 0 if np.matmul(weightAnnealMountain, observation) < 0 else 2
            else:
                state = get_state_as_string(assign_bins(observation, bins))
                action = int(stateActionPairs[state])

            observation, reward, done, info = env.step(action)

            if done:
                #print("\nEpisode  finished  at  t = " + str(t+1))
                break
        env.close()
        scores.append(t+1)
    averages.append(float(sum(scores))/float(len(scores)))
    print("\nAverage Score: " + str(float(sum(scores))/float(len(scores))))

if HillMountain or AnnealMountain:
    print("\nMean Score: " + str(-np.mean(averages)))
    print("Score Std Dev: " + str(np.std(averages)))
else:
    print("\nMean Score: " + str(np.mean(averages)))
    print("Score Std Dev: " + str(np.std(averages)))

