
#This Code Tests a Deep Q learning Policy in the CartPole Environment.
import gym
import numpy as np
from keras.models import load_model

model = load_model ('DQ-500.h5')
#model = load_model ('CP-DQ-HPOLE.h5')
#model = load_model ('DCP-10TAU.h5')
    
def act (model, state):    
    q_values = model.predict (state)
    return np.argmax (q_values[0])

#envs = [(gym.make('cp-mars_gravity-v0'), "Mars Gravity"), ('cp-half_pole_len-v0', 'Half Pole Length'), 
		#('cp-double-pole-mass-v0', 'Double Pole Mass'), ('cp-double-force-v0', "Double Force"),
		#('cp-double-cart-mass-v0', "Double Cart Mass")]

#for environment in envs:
n_episodes = 20
curenv = "test"
env = gym.make ('CartPole-v1')
env._max_episode_steps = 500
scores = []
for i_episode in range(n_episodes):
    for t in range(300):
        if t == 0:
            observation = env.reset()  # initialize to semi-random start state

        state = np.reshape (observation, [1, 4])

        action = act (model, state)
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    scores.append(t+1)
print "Environment: " + curenv
avg = float(sum(scores))/float(len(scores))
print "\nAverage Score: " + str(avg)
print "\nSTD DEV: " + str ( (sum ([((i - avg) ** 2) / float (len (scores)) for i in scores])) ** 0.5)