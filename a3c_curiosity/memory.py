import numpy as np
import torch

class Memory:
    def __init__(self):
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.states = []
        self.next_states = []
        self.actions = []

    def remember(self, state, action, next_state, reward, value, log_prob):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear_memory(self):
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.states = []
        self.next_states = []
        self.actions = []

    def sample_memory(self):
        return self.states, self.actions, self.next_states, self.rewards, self.values, self.log_probs



class Memory_mine:
    def __init__(self, t_max):
        self.t_max = t_max
        self.mem_idx = 0
        self.values = np.zeros(t_max)
        self.log_probs = np.zeros(t_max)
        self.rewards = np.zeros(t_max)

    def clear_memory(self):
        self.values = np.zeros(self.t_max)
        self.log_probs = np.zeros(self.t_max)
        self.rewards = np.zeros(self.t_max)

    def add_to_memory(self, value, log_prob, reward):
        self.values[self.mem_idx] = value
        self.log_probs[self.mem_idx] = log_prob
        self.rewards[self.mem_idx] = reward
        self.mem_idx += 1
        if self.mem_idx % self.t_max == 0:
            self.mem_idx = 0

    def retrieve(self):
        return self.values, self.log_probs, self.rewards