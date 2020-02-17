from functions import softmax
from functions import CEE
import numpy as np
import sys, os
sys.path.append(os.pardir)

class simple_net:

    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x,self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax.softmax(z)
        loss = CEE.cross_entropy_error(y, t)

        return loss


