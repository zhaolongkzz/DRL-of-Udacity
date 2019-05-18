import numpy as np
import random
from collections import defaultdict


class Agent:

    def __init__(self, nA=6, alpha=0.09, eps=0.01, gamma=0.8):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.alpha = alpha
        self.epsilon = eps
        self.gamma = gamma
        
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))


    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.epsilon = min (self.epsilon / i_episode , 0.005)
        probs = [self.epsilon/self.nA]*self.nA
        act = np.argmax(self.Q[state])
        probs[act] = 1- self.epsilon + self.epsilon/self.nA
        return np.random.choice(np.arange(self.nA), p=probs)
    

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
#         using expected sarsa
#         current = self.Q[state][action]
#         policy_s = np.ones(self.nA) * self.epsilon / self.nA
#         policy_s[np.argmax(self.Q[next_state])] = 1 - self.epsilon + (self.epsilon / self.nA)
#         target = np.dot(self.Q[next_state], policy_s) + reward
#         new_Q = current + (self.alpha * (target - current))
#         self.Q[state][action] = new_Q
        
#         using Q-learning
        current = self.Q[state][action]
        next_Q = np.max(self.Q[next_state]) if next_state is not None else 0
        target = reward + (self.gamma * next_Q)  
        new_Q = current + (self.alpha * (target - current))
        self.Q[state][action] = new_Q

