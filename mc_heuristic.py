##basic heuristic, go left as high as you can, then go right as high as you can

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

minimum_epsilon = 0.5
minimum_eta = 0.5

np.random.seed(1)

successful = []
unsuccessful = []
means = []   

for episode in range(1,500*50):
    eta = 0.05
    obs = env.reset()
    #print "first obs"
    #print obs
    if obs[0] > -0.5:
        prev_action = 2 
    elif obs[0] < -0.5:
        prev_action = 0
    else:
        prev_action = 1
    prev_vel = obs[1]
    prev_actions = []
    prev_vels = []
    prev_poss = []
    abs_max_height = 0
    for step in range(200):
        #if the velocity of the cart is zero, see which direction it was previously going and then switch it
        action_ = prev_action
        if prev_vel < 1e-4: #this is the first iteration, or the car has stopped moving
            if obs[0] > -0.5:
                if prev_action == 2:
                    action_ = 0
            elif obs[0] < -0.5:
                if prev_action == 0:
                    action_ = 2
        next_obs, reward, done, info = env.step(action_)
        prev_actions.append(action_)
        prev_vels.append(next_obs[1])
        prev_poss.append(obs[0])
        prev_vel = obs[1]
        prev_pos = obs[0]
        if(done):
            break
        obs = next_obs
        prev_action = action_
        
    if(step<199):
        result = "Successful"
        successful.append(step)
        print("Episode {} : {}".format(episode,result))

    else:
        result = "Unsuccessful"
        unsuccessful.append(step)
        print("Episode {} : {}".format(episode,result))
    if (episode % 500) == 0:
        sum = 0
        for x in range(len(successful)):
            sum += successful[x]
        for x in range(len(unsuccessful)):
            sum += unsuccessful[x]
        mean = ((sum)/(len(successful) + len(unsuccessful)))
        means.append(mean)

##PRINTING MEANS AND STD DEVIATIONS##
print "means"
print means
print "mean"
sum = 0
for x in range(len(means)):
    sum += means[x]
print sum/len(means)
print "std_dev"
print np.std(np.asarray(means))
