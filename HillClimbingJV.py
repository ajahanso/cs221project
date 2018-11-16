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
timesOver = 1 #How many times over the data
n_episodes = 50 #take average score over 5 episodes and then make decision about reject or accept action
numActions = 200
visualization = False
scores = []
stateActionPairs = {}
bestAverage = -float('Inf')
exportFilename = "Hill Climbing.txt"
#inputFilename = "Random Policy.txt"
#inputFilename = "Policy Simple Q.txt"
inputFilename = "Heuristic Policy.txt"



df = pd.read_csv(inputFilename, dtype=str)
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
for time in range(timesOver):
    print("Time: " + str(time))
    for actionChange in range(numActions):
        #Change action
        tempAct = random.randint(0, len(stateActionPairs))
        if stateActionPairs[df.at[tempAct, 's']] == 1:
            stateActionPairs[df.at[tempAct, 's']] = 0
        else:
            stateActionPairs[df.at[tempAct, 's']] = 1
        #print("Action Investigated: " + str(tempAct))
        averages = []
        #See how the change does over multiple episodes
        for i_episode in range(n_episodes):
            for t in range(1000):
                if t == 0:
                    observation = env.reset()  # initialize to semi-random start state

                if visualization:
                    env.render() #Graphics on or off

                state = get_state_as_string(assign_bins(observation, bins))
                action = int(stateActionPairs[state])
                observation, reward, done, info = env.step(action)

                if done:
                    #print("\nEpisode  finished  at  t = " + str(t+1))
                    break
            env.close()
            scores.append(t+1)
        averages.append(float(sum(scores))/float(len(scores)))
        if float(sum(scores))/float(len(scores)) > bestAverage:
            bestAverage = float(sum(scores))/float(len(scores))
            print("Action Changed")
        else:
            #change action back
            if stateActionPairs[df.at[actionChange, 's']] == 1:
                stateActionPairs[df.at[actionChange, 's']] = 0
            else:
                stateActionPairs[df.at[actionChange, 's']] = 1

        #print("\nAverage Score: " + str(float(sum(scores))/float(len(scores))))


#print("Mean Score: " + str(np.mean(averages)))
#print("Score Std Dev: " + str(np.std(averages)))



###OUTPUT LEARNED POLICY TO FILE###
file = open(exportFilename, "w")
file.write("s,a") #CSV file: state,action
for state in stateActionPairs:
    act = stateActionPairs[state]
    file.write("\n" + str(state) + "," + str(act)) #CSV file: state,action
file.close()

print("DONE!")