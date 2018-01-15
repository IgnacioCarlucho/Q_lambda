import gym
import numpy as np
import random
import time


# Generate a frozen lake environment without skid
# this is easier to solve for a tabular RL 
from gym.envs.registration import register, spec
#env = gym.make('FrozenLake-v0')
MY_ENV_NAME='FrozenLakeNonskid8x8-v0'
try:
    spec(MY_ENV_NAME)
except:
    register(
        id=MY_ENV_NAME,
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        timestep_limit=100,
        reward_threshold=0.8196, # optimum = .8196
    )
env = gym.make(MY_ENV_NAME)
# Variables
num_action = env.action_space.n
num_states = env.observation_space.n
epochs = 2000
epsilon = 0.99
gamma = 0.9 # discount factor
value_lambda = 0.8
alpha = 0.1 # learning rate
Q_table = np.zeros((num_states,num_action))
E_table = np.zeros((num_states,num_action))

# variables to plot
wins = 0
losses = 0
step = 0
reward = 0.
for i in range(epochs):
    #print(i,step ,round(epsilon,3), wins, 'last rewrd', reward )
    state = env.reset()
    done = False
    reward = 0
    epsilon = np.maximum(epsilon - 0.001, 0.)
    step = 0
    while not done:
        step = step +1
        
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,num_action)
        else: #choose best action from Q(s,a) values
            action = np.argmax(Q_table[state])
        #print(Q_table[state])
        #print(action)
        #time.sleep(5)
        new_state, reward, done, info = env.step(action)
        #print(i, step, new_state, reward)

        if not done: # Non-terminal state.
           next_q = np.max(Q_table[new_state])
           target = reward + ( gamma * next_q ) - Q_table[state][action]
        else:
            if reward == 1.:
                target = reward + 9.
            else:
            	target = reward
        #print(E_table)
        E_table[state][action] = E_table[state][action] + 1.
        
        #print(Q_table)
        #time.sleep(10)
        for y in range(num_action):
            for x in range(num_states):   
                Q_table[x][y] = Q_table[x][y] + (alpha*target*E_table[x][y])
                E_table[x][y] = gamma*value_lambda*E_table[x][y] 
        #print(Q_table)
        #time.sleep(1)
        state = new_state
        if done:
            if new_state == 15:
                wins = wins +1
            else:
            	losses = losses +1

        
print(Q_table)
print('e', E_table)
print('wins', wins, 'losses', losses, 'efficiency', round(100.*wins/losses,2))


