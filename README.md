# cs221project
cs221 Fall 2018 Project

Reinforcement Learning in OpenAI Gym
Alison Jahansouz, Jackson Lallas, Joseph Vincent

cartpole.py and mountain_car.py are copies of the base environments from OpenAI Gym of CartPole and Mountain Car respectively. In our testing, we navigated to the directories where gym was installed and manually modified parameters of the problem. So if we wanted to test CartPole with a .25 pole length, we would navigate to where pip installed cartpole.py, change the self.length variable, and then save the environment.



DeepQ_CP.py and ExperimentsDQ.py are the files for creating deep q-learning policies and testing them in the cartpole problem. To use, simply invoke the commands:

python DeepQ_CP.py

python ExperimentsDQ.py

In each file, there is a name field for what the policy trained should be saved as and which policy to load for testing on the environment. Be sure when using them that the saved name and loaded policy match the experiment to be performed. For instance, if the default policy was saved as "DQ-DEF" and you want to test the default policy on an updated cart mass environment, first change the environment as described earlier and then make sure that the "DQ-DEF" file is loaded in ExperimentsDQ.py

