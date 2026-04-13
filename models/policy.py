import random

class PolicyModel:
    def __init__(self):
        self.safety_bias = 0.5 # probability of safe response
        
    def generate(self, prompt):
        if random.random() < self.safety_bias:
            return "SAFE_RESPONSE"
        else:
            return "UNSAFE_RESPONSE"
        
    def update(self, reward):
        # simple update rule
        self.safety_bias += 0.1 * reward
        self.safety_bias = max(0, min(1, self.safety_bias))