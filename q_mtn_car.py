import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

minimum_epsilon = 0.5
minimum_eta = 0.5

np.random.seed(1)

#Helper Function

def getAction(eta, obs, partition, QTable):
    if(np.random.uniform() <= eta):
        action = env.action_space.sample()
    else:
        action = getBestAction(obs, partition, QTable)
    return action

def getReward(prev_obs,curr_obs,action):
    if(curr_obs[0] - prev_obs[0]) >= 0:
        return (curr_obs[1] - prev_obs[1])
    else:
        return -1 * (curr_obs[1]-prev_obs[1])

def getState(obs, partition):
    obs_dict = env.observation_space.__dict__ #what does this do
    #print obs_dict
    max_obs = obs_dict["high"]
    min_obs = obs_dict["low"]
    obs_range = max_obs - min_obs

    
    state=[]
    for index, value in enumerate(obs): #for each observation
        step = obs_range[index]/partition[index]
        threshold = min_obs[index] + step
        state_code = 0
        for _ in range(1,partition[index]):
            if(value <= threshold):
                break
            else:
                state_code += 1
                threshold += step
        state.append(state_code)
    return state
        
def generateQTable(shape):
    return np.zeros(np.prod(shape)).reshape(*shape)

def getQ(state, action, QTable):
    return QTable[state[0]][state[1]][action[0]]

def setQ(new_value,state, action, QTable):
    QTable[state[0]][state[1]][action[0]] = new_value

def getMaxQ(state, QTable):
    result = QTable
    for element in state:
        result = result[element]
    return np.max(result)

def getBestAction(obs, partition, QTable): #[100,2]
    state = getState(obs, partition)
    result = QTable
    for element in state:
        result = result[element]
    return list(result).index(np.max(result))

def updateQTable(obs, next_obs, action, partition, epsilon , discount_rate, reward, QTable):
    state = getState(obs, partition)
    next_state = getState(next_obs, partition)
    current_q = getQ(state, action, QTable)
    max_future_q = getMaxQ(next_state, QTable)
    setQ(epsilon *((-1*(current_q))+ (discount_rate * max_future_q) + getReward(obs,next_obs,action)), state, action, QTable)
    return epsilon *((-1*(current_q))+ (discount_rate * max_future_q) + getReward(obs,next_obs,action))

QTable = generateQTable([100,2,3])

successful = []
unsuccessful = []
means = []

for episode in range(1,500*50):
    eta = 0.05
    obs = env.reset()
    for step in range(200):
        action_= getAction(eta, obs, [100,2], QTable)
        next_obs, reward, done, info = env.step(action_)
        #env.render()
        if(done):
            break
        else:
            reward2 = updateQTable(obs, next_obs, [action_], [100,2], 0.5, 0.5, reward, QTable)
            obs = next_obs

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



print "means"
print means
print "mean"
sum = 0
for x in range(len(means)):
    sum += means[x]
print sum/len(means)

print "std_dev"
print np.std(np.asarray(means))

print "original stuff"

sum = 0
for x in range(len(successful)):
    sum += successful[x]
sum2 = 0
for x in range(len(unsuccessful)):
    sum2 += unsuccessful[x]
#print successful + unsuccessful
print "mean total"
print_this = ((sum + sum2)/(len(successful) + len(unsuccessful)))
print print_this
