'''
This code implements basic Q-learning for the CartPole_V0 environment
'''

import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import random

# returns maximum value in a dictionary with associated key
def maxDict(d):
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v


# turns state into a string. ex) 1234 -> '1234'
# an allStates list is easier to generate with string states
def get_state_as_string(state):
    string_state = ''.join(str(int(e)) for e in state)
    return string_state


# Puts continuous observations in discrete bins
def assign_bins(observation, bins):
    state = [0]*4
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    return state


# Goes through Q-learning algorithm for one episode
def oneEpisode(bins, Q, eta, gamma, epsilon, animate):
    observation = env.reset() #initialize to semi-random start state
    state = get_state_as_string(assign_bins(observation, bins))
    done = False
    totalReward = 0
    t = 0  # number of time-steps

    while not done:
        if animate:
            env.render()
        t += 1
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()  # epsilon greedy
        else:
            action = maxDict(Q[state])[0]  # returns action associated with highest Q value for that state

        observation, reward, done, info = env.step(action) #results of action
        newState = get_state_as_string(assign_bins(observation, bins))
        totalReward += reward

        if done and t < 200:
            reward = -300 #Harsh penalty for ending before timeout state

        newAction, maxQ = maxDict(Q[newState])

        # Q learning formula
        Q[state][action] -= eta * (Q[state][action] - (reward + gamma * maxQ))
        state = newState
        action = newAction

    env.close()
    return totalReward, t



####################################################################################
##################################BEGIN PROGRAM#####################################
####################################################################################

env = gym.make('CartPole-v0')

###HYPERPARAMETERS###
maxStates = 10**4  # function of bin discretization
gamma = 0.9        # discount factor
eta = 0.01         # step size
numEpisodes = 500  # episodes to train over
animate = False    # boolean for showing animation


###INITIALIZE LIST OF ALL POSSIBLE STATES###
all_states = []
for i in range(maxStates):
    all_states.append(str(i).zfill(4))


###INITIALIZE Q Dictionary###
Q = {}
for state in all_states:
    Q[state] = {}
    for action in range(env.action_space.n):
        Q[state][action] = 0


###CREATE DISCRETE BINS###
bins = np.zeros((4, 10))              # 10 discretizations for each variable
bins[0] = np.linspace(-4.8, 4.8, 10)  # observation[0]: cart position: -4.8 - 4.8
bins[1] = np.linspace(-5, 5, 10)      # observation[1]: cart velocity: -inf - inf
bins[2] = np.linspace(-.4, .4, 10)    # observation[2]: pole angle: -41.8 - 41.8
bins[3] = np.linspace(-5, 5, 10)      # observation[3]: pole velocity: -inf - inf


###TRAIN & EVALUATE###
episodeLengths = []
episodeRewards = []
iterSolved = 0
for n in range(numEpisodes):
    epsilon = 1.0 / np.sqrt(n + 1) # epsilon greedy policy

    episode_reward, episode_length = oneEpisode(bins, Q, eta, gamma, epsilon, animate)

    episodeLengths.append(episode_length)
    episodeRewards.append(episode_reward)
    # print("\nTime: " + str(episode_length))
    # print("N = " + str(n))

    total = 0
    if len(episodeRewards) > 19 and iterSolved == 0:
        for i in range(len(episodeRewards)-20, len(episodeRewards)):
            total += episodeRewards[i]
        if total/20. >= 195.:
            iterSolved = n

print("# iterations to solve: " + str(iterSolved))


###PLOTTING###
windowSize = 20
N = len(episodeRewards)
running_avg = [None]*N
for t in range(N):
    if t > 18:
        running_avg[t] = np.mean(episodeRewards[(t + 1 - windowSize):(t + 1)])
    else:
        running_avg[t] = np.mean(episodeRewards[0:(t + 1)])
plt.plot(running_avg)
plt.title("20-Episode Average Score")
plt.xlabel("Episode")
plt.ylabel("Average Score")
plt.show()


###OUTPUT LEARNED POLICY TO FILE###
file = open("Policy Simple Q.txt", "w")
for state in all_states:
    highest = -float("Inf")
    for action in Q[state]:
        if Q[state][action] > highest:
            highest = Q[state][action]
            act = action
        elif Q[state][action] == highest: #If state not visited, tie break with random
            act = random.randint(0, env.action_space.n - 1)
    file.write(str(state) + "," + str(act) + "\n") #CSV file: state,action
file.close()
