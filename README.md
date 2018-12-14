# cs221project
cs221 Fall 2018 Project

Reinforcement Learning in OpenAI Gym
Alison Jahansouz, Jackson Lallas, Joseph Vincent

cartpole.py and mountain_car.py are copies of the base environments from OpenAI Gym of CartPole and Mountain Car respectively. In our testing, we navigated to the directories where gym was installed and manually modified parameters of the problem. So if we wanted to test CartPole with a .25 pole length, we would navigate to where pip installed cartpole.py, change the self.length variable, and then save the environment.



DeepQ_CP.py and ExperimentsDQ.py are the files for creating deep q-learning policies and testing them in the cartpole problem. To use, simply invoke the commands:


python DeepQ_CP.py

python ExperimentsDQ.py

In each file, there is a name field for what the policy trained should be saved as and which policy to load for testing on the environment. Be sure when using them that the saved name and loaded policy match the experiment to be performed. For instance, if the default policy was saved as "DQ-DEF" and you want to test the default policy on an updated cart mass environment, first change the environment as described earlier and then make sure that the "DQ-DEF" file is loaded in ExperimentsDQ.py


CartPoleHillClimbing.py output a weight vector which was updated by making random changes to a previous weight vector and either accepting or rejecting the changes. This weight vector is multiplied by the state vector and if the product is positive, the action is +1, and -1 otherwise. 

CartPoleSimulatedAnnealing.py outputs a similar weight vector as CartPoleHillClimbing.py. Simulated annealing methods are used instead of basic hill climbing methods. 

CartPole_SimpleQ.py outputs a .csv file which represents a policy for each state in the discretization. 

MountainCarHillClimbing.py is the same as CartPoleHillClimbing.py except modified for the MountainCar environment.

MountainCarSimulatedAnnealing.py is the same as CartPoleSimulatedAnnealing.py except modified for the MountainCar environment.

For each of these files, we invoked it with the command "python x" where x is any of the algorithm files. If we wanted to save different policies, we would change the name of the file output within the algorithm's code. 

PolicyTesting.py is used to test polices of different algorithms. Either a .csv file is imported, or a weight vector is used. cartpole.py and mountain_car.py must be modified for changes to happen in testing using PolicyTesting.py. To run policy testing, simply use the command python PolicyTesting.py and make sure that the correct file / weight vector is loaded. We manually changed the PolicyTesting.py file each time we wanted to test a different policy file / weight vector.
