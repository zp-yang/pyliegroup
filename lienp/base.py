import numpy as np

EPS = 1e-7

def wrap(x):
    return np.where(np.abs(x) >= np.pi, (x + np.pi) % (2 * np.pi) - np.pi, x)

def wedge2(w):
    pass

def vee2(wx):
    pass

def wedge3(w):
    w_ = np.squeeze(w)
    wx = np.array([
        [0, -w_[2], w_[1]],
        [w_[2], 0, -w_[0]],
        [-w_[1], w_[0], 0]
    ])
    return wx

def vee3(wx):
    w = np.array([wx[2,1], wx[0,2], wx[1,0]])
    return w

class LieAlg:
    def __init__(self, param=None):
        self.param = param
        pass
    
    @property
    def exp(self):
        pass

    @classmethod
    def log(cls, grp):
        pass

class LieGrp:
    def __init__(self, param=None):
        self.param = param
        pass
    
    @classmethod
    def exp(cls, alg):
        pass

    @property
    def log(self):
        pass
