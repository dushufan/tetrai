from collections import deque
from numpy import random


class ReplayMemory:
    """ circular buffer stores completed game moves for retraining """

    def __init__(self, size=20000):
        self.memories = deque([], maxlen=size)
        self.mem_size = size

    def remember(self, experience):
        """
        experience is a 4-tuple
        (current state, new state, reward, game over)
        """
        self.memories.append(experience)

    def sample(self, size=512):
        """ returns a random sample of the current memories """
        samples = []
        if len(self.memories) == self.mem_size:
            for _ in range(size):
                samples.append(self.memories[random.randint(0, self.mem_size)])
        return samples
