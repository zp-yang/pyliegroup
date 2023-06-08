from .base import *
from enum import Enum
class Repr(Enum):
    EULER = 0
    DCM = 1
    AXIS = 2
    MRP = 3
    QUAT = 4

class so3(LieAlg):
    def __init__(self, param=None, param_repr=Repr.EULER):
        if param is not None:
            super().__init__(param=param)
            self.param = param
            self.param_repr = param_repr
            
        else:
            """Intialize to zeros"""
            self.param = np.zeros(3)
            self.param_repr = Repr.EULER

    @property
    def wedge(self):
        if self.param_repr == Repr.AXIS:
            return wedge3(self.param)
        elif self.param_repr == Repr.EULER:
            return wedge3(self.log(SO3.from_euler(self.param)))

    @property
    def exp(self):
        if self.param_repr == Repr.EULER:
            return SO3.from_euler(self.param)
        elif self.param_repr == Repr.AXIS:
            w = self.param
            theta = np.linalg.norm(w)
            if np.abs(theta) > EPS:
                C1 = np.sin(theta)/theta
                C2 = (1 - np.cos(theta))/theta**2
            else:
                C1 = 1 - theta**2/6 + theta**4/120 - theta**6/5040
                C2 = 1/2- theta**2/24 + theta**4/720 - theta**5/40320
            wx = self.wedge
            R = np.eye(3) + C1 * wx + C2 * wx @ wx
            return R
        else:
            raise NotImplementedError ("")
    
    @classmethod
    def log(cls, R):
        theta = np.arccos((np.trace(R) - 1) / 2)
        if np.abs(theta) > EPS:
            C3 = theta/(2*np.sin(theta))
        else:
            C3 = 0.5 + theta**2/12 + 7*theta**4/720
        return vee3(C3 * (R - R.T))
    
    @property
    def ad(self):
        raise NotImplementedError("so3 ad not implemented")

class SO3(LieGrp):
    def __init__(self, param=None, param_repr=Repr.DCM):
        if param is not None:
            super().__init__(param=param)
            self.param = param
            self.param_repr = param_repr

            if param_repr == Repr.DCM:
                assert param.shape == (3,3)
                self.R = param
            elif param_repr == Repr.EULER:
                assert param.shape == (3,) or param.shape == (3,1)
                self.R = SO3.from_euler(param)
            elif param_repr == Repr.QUAT:
                assert param.shape == (4,) or param.shape == (4,1)
                self.R = SO3.from_quat(param)
            else:
                raise NotImplementedError("Initializing from other representation is not implemented")

        else:
            """Intialize to identities"""
            self.param = np.eye(3)
            self.alg = np.zeros(3)
            self.param_repr = Repr.DCM
            self.R = np.eye(3)
    
    def __repr__(self) -> str:
        return f"R: {self.R}, param: {self.param}, param_repr:{self.param_repr}"
    
    @property
    def Ad(self):
        return self.R
    
    @property
    def inv(self):
        return self.R.T
    
    # @property
    # def vee(self, wx):
    #     return self.alg

    @classmethod
    def exp(cls, w):
        theta = np.linalg.norm(w)
        if np.abs(theta) > EPS:
            C1 = np.sin(theta)/theta
            C2 = (1 - np.cos(theta))/theta**2
        else:
            C1 = 1 - theta**2/6 + theta**4/120 - theta**6/5040
            C2 = 1/2- theta**2/24 + theta**4/720 - theta**5/40320
        wx = wedge3(w)
        R = np.eye(3) + C1 * wx + C2 * wx @ wx
        return R
    
    @property
    def log(self):
        theta = np.arccos((np.trace(self.R) - 1) / 2)
        if np.abs(theta) > EPS:
            C3 = theta/(2*np.sin(theta))
        else:
            C3 = 0.5 + theta**2/12 + 7*theta**4/720
        return vee3(C3 * (self.R - self.R.T))
    
    @classmethod
    def from_euler(cls, e):
        phi = e[0]
        theta = e[1]
        psi = e[2]
        
        R1 = np.array([
            [1, 0, 0],
            [0, np.cos(phi), np.sin(phi)],
            [0, -np.sin(phi), np.cos(phi)],
        ])
        R2 = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ])
        R3 = np.array([
            [np.cos(psi), np.sin(psi), 0],
            [-np.sin(psi), np.cos(psi), 0],
            [0, 0 , 1]
        ])
        dcm = R1 @ R2 @ R3
        return dcm.T
    
    @classmethod
    def from_quat(cls, q):
        assert q.shape == (4, 1) or q.shape == (4,)
        R = np.zeros((3,3))
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        aa = a * a
        ab = a * b
        ac = a * c
        ad = a * d
        bb = b * b
        bc = b * c
        bd = b * d
        cc = c * c
        cd = c * d
        dd = d * d
        R[0, 0] = aa + bb - cc - dd
        R[0, 1] = 2 * (bc - ad)
        R[0, 2] = 2 * (bd + ac)
        R[1, 0] = 2 * (bc + ad)
        R[1, 1] = aa + cc - bb - dd
        R[1, 2] = 2 * (cd - ab)
        R[2, 0] = 2 * (bd - ac)
        R[2, 1] = 2 * (cd + ab)
        R[2, 2] = aa + dd - bb - cc
        return R
    
    @classmethod
    def quat_to_euler(cls, q):
        assert q.shape == (4, 1) or q.shape == (4,)
        e = np.zeros((3,))
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        e[0] = np.arctan2(2 * (a * b + c * d), 1 - 2 * (b**2 + c**2))
        e[1] = np.arcsin(2 * (a * c - d * b))
        e[2] = np.arctan2(2 * (a * d + b * c), 1 - 2 * (c**2 + d**2))
        return e
    
    @classmethod
    def euler_to_quat(cls, e):
        assert e.shape == (3, 1) or e.shape == (3,)
        q = np.zeros((4,))
        cosPhi_2 = np.cos(e[0] / 2)
        cosTheta_2 = np.cos(e[1] / 2)
        cosPsi_2 = np.cos(e[2] / 2)
        sinPhi_2 = np.sin(e[0] / 2)
        sinTheta_2 = np.sin(e[1] / 2)
        sinPsi_2 = np.sin(e[2] / 2)
        q[0] = cosPhi_2 * cosTheta_2 * cosPsi_2 + sinPhi_2 * sinTheta_2 * sinPsi_2
        q[1] = sinPhi_2 * cosTheta_2 * cosPsi_2 - cosPhi_2 * sinTheta_2 * sinPsi_2
        q[2] = cosPhi_2 * sinTheta_2 * cosPsi_2 + sinPhi_2 * cosTheta_2 * sinPsi_2
        q[3] = cosPhi_2 * cosTheta_2 * sinPsi_2 - sinPhi_2 * sinTheta_2 * cosPsi_2
        return q

    @classmethod
    def to_euler(cls, R):
        assert R.shape == (3,3)
        return cls.quat_to_euler(cls.to_quat(R))

    @classmethod
    def to_quat(cls, R):
        assert R.shape == (3, 3)
        b1 = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        b2 = 0.5 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        b3 = 0.5 * np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        b4 = 0.5 * np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])

        q1 = np.zeros((4,))
        q1[0] = b1
        q1[1] = (R[2, 1] - R[1, 2]) / (4 * b1)
        q1[2] = (R[0, 2] - R[2, 0]) / (4 * b1)
        q1[3] = (R[1, 0] - R[0, 1]) / (4 * b1)

        q2 = np.zeros((4,))
        q2[0] = (R[2, 1] - R[1, 2]) / (4 * b2)
        q2[1] = b2
        q2[2] = (R[0, 1] + R[1, 0]) / (4 * b2)
        q2[3] = (R[0, 2] + R[2, 0]) / (4 * b2)

        q3 = np.zeros((4,))
        q3[0] = (R[0, 2] - R[2, 0]) / (4 * b3)
        q3[1] = (R[0, 1] + R[1, 0]) / (4 * b3)
        q3[2] = b3
        q3[3] = (R[1, 2] + R[2, 1]) / (4 * b3)

        q4 = np.zeros((4,))
        q4[0] = (R[1, 0] - R[0, 1]) / (4 * b4)
        q4[1] = (R[0, 2] + R[2, 0]) / (4 * b4)
        q4[2] = (R[1, 2] + R[2, 1]) / (4 * b4)
        q4[3] = b4

        q = np.where(
            np.trace(R) > 0,
            q1,
            np.where(
                np.logical_and(R[0, 0] > R[1, 1], R[0, 0] > R[2, 2]),
                q2,
                np.where(R[1, 1] > R[2, 2], q3, q4),
            ),
        )
        return q
