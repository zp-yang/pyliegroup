from .base import *
from .SO3 import *

class se3(LieAlg):
    def __init__(self, param=None):
        super().__init__(param)
        assert param.shape == (6,) or param.shape == (6,1)
        if param.shape == (6,1):
            param = np.squeeze(param)
        self.u = param[0:3]
        self.w = param[3:6]

    @property
    def exp(self):
        R = np.array(SO3.exp(self.w))
        t = np.array(_V(self.w) @ self.u)
        c = -R.T @ t
        return SE3(R=R, c=c)
    
    @classmethod
    def log(cls, G):
        pass


class SE3(LieGrp):
    """
    SE3 parameter should be a (6,) array pose [pos, rpy],
    or construct with R and c
    """
    def __init__(self, param=None, R=None, c=None):
        super().__init__(param)
        if param is not None:
            assert param.shape == (6,) or param.shape == (6,1)
            self.c = np.array(param[0:3]) # translation
            self.R = np.array(SO3.from_euler(param[3:6])) # Euler to DCM
            self.t = -self.R @ self.c
        elif R is not None and c is not None:
            self.R = R
            self.c = np.squeeze(c)
            self.t = -R @ c
        else:
            raise NotImplementedError("Intialization error")
        
        self.M = np.block([
            [self.R, self.t.reshape(3,1)],
            [np.zeros((1,3)), np.array(1)]
        ])
    
    def  __repr__(self) -> str:
        return repr(self.M)
    
    def __add__(self, other):
        raise NotImplementedError("Addition is not defined in SE3")
    
    def __mul__(self, other):
        raise NotImplementedError("Elementwise multiplication is not defined in SE3")

    def __matmul__(self, other):
        if type(other) is SE3:
            R = self.R @ other.R
            t = self.R @ other.t + self.t
            # param = np.hstack([np.array(f_dcm2euler(R)).reshape(3,), t.reshape(3,)])
            return SE3(R=R, c=-R.T @ t)
        elif type(other) is np.ndarray:
            return self.M @ other

    @property
    def shape(self):
        return (4,4)

    @property
    def inv(self):
        # M_inv = np.block([
        #     [self.R.T, (-self.R.T @ self.t).reshape(3,1)],
        #     [np.zeros((1,3)), 1]
        # ])
        # return M_inv
        return SE3(R=self.R.T, c=self.t)
        

    @property
    def log(self):
        return se3(self.R, self.c)

    @property
    def Ad(self):
        tx = np.array([
            [0, -self.t[2], self.t[1]],
            [self.t[2], 0, -self.t[0]],
            [-self.t[1], self.t[0], 0]
        ])
        X = np.block([
            [self.R, tx @ self.R],
            [np.zeros((3,3)), self.R]
        ])
        return X
    
    @property
    def Ad_inv(self):
        tx = np.array([
            [0, -self.t[2], self.t[1]],
            [self.t[2], 0, -self.t[0]],
            [-self.t[1], self.t[0], 0]
        ])
        X = np.block([
            [self.R.T, -self.R.T @ tx],
            [np.zeros((3,3)), self.R.T]
        ])
        return X
    
    @classmethod
    def exp(cls, g):
        u = np.squeeze(g[0:3])
        w = np.squeeze(g[3:6])
        R = np.array(SO3.exp(w))
        # e = np.array(f_dcm2euler(R))
        t = np.array(_V(w) @ u)
        c = -R.T @ t
        return SE3(R=R, c=c)
    
    @property
    def log(self):
        w = np.array(SO3.log(self.R))
        u = _V_inv(w) @ self.t
        return se3(np.hstack([u, w]))
    
    def get_param(self):
        if self.param is not None:
            return self.param
        else:
            return np.hstack([self.c, SO3.to_euler(self.R)])

   
def _V(w):
    theta = np.linalg.norm(w)
    wx = wedge3(w)
    V = np.eye(3) + B(theta) * wx + C(theta) * wx @ wx
    return V

def _V_inv(w):
    theta = np.linalg.norm(w)
    wx = wedge3(w)
    V_inv = np.eye(3) - 0.5 * wx + 1/(theta**2) * (1 - A(theta)/(2*B(theta))) * wx @ wx
    return V_inv
    
A = lambda x: np.where(np.abs(x)<EPS, 1 - x**2 / 6 + x**4 / 120, np.sin(x)/x)
B = lambda x: np.where(np.abs(x)<EPS, 0.5 - x**2 / 24 + x**4 / 720, (1 - np.cos(x))/x)
C = lambda x: np.where(np.abs(x)<EPS, 1/6 - x**2 / 120 + x**4 / 5040, (1 - A(x)) / x**2)