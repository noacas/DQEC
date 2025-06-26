"""
Implementation of "Deep Quantum Error Correction" (DQEC), AAAI24
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from functools import lru_cache

import numpy as np
import torch
from scipy.sparse import hstack, kron, eye, csr_matrix, block_diag
import itertools
import scipy.linalg


class Code:
    '''
    Base class for quantum error correction codes.
    '''
    def __init__(self, code_type, code_L, noise_type='depolarization'):
        self.code_type = code_type
        H, Lx = eval(f'Get_{code_type}_Code')(code_L, full_H=noise_type)
        self.logic_matrix = torch.from_numpy(Lx).long()
        self.pc_matrix = torch.from_numpy(H).long()
        self.n = self.pc_matrix.shape[1]
        self.k = self.n - self.pc_matrix.shape[0]


class ToricCode:
    '''
    From https://github.com/Krastanov/neural-decoder/
        Lattice:
        X00--Q00--X01--Q01--X02...
         |         |         |
        Q10  Z00  Q11  Z01  Q12
         |         |         |
        X10--Q20--X11--Q21--X12...
         .         .         .
    '''
    def __init__(self, L):
        '''Toric code of ``2 L**2`` physical qubits and distance ``L``.'''
        self.L = L
        self.Xflips = np.zeros((2*L,L), dtype=np.dtype('b')) # qubits where an X error occured
        self.Zflips = np.zeros((2*L,L), dtype=np.dtype('b')) # qubits where a  Z error occured
        self._Xstab = np.empty((L,L), dtype=np.dtype('b'))
        self._Zstab = np.empty((L,L), dtype=np.dtype('b'))

    @property
    def flatXflips2Zstab(self):
        L = self.L
        _flatXflips2Zstab = np.zeros((L**2, 2*L**2), dtype=np.dtype('b'))
        for i, j in itertools.product(range(L),range(L)):
            _flatXflips2Zstab[i*L+j, (2*i  )%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+1)%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+2)%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+1)%(2*L)*L+(j+1)%L] = 1
        return _flatXflips2Zstab

    @property
    def flatZflips2Xstab(self):
        L = self.L
        _flatZflips2Xstab = np.zeros((L**2, 2*L**2), dtype=np.dtype('b'))
        for i, j in itertools.product(range(L),range(L)):
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+1)%(2*L)*L+(j+1)%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+2)%(2*L)*L+(j  )%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+3)%(2*L)*L+(j+1)%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+2)%(2*L)*L+(j+1)%L] = 1
        return _flatZflips2Xstab

    @property
    def flatXflips2Zerr(self):
        L = self.L
        _flatXflips2Zerr = np.zeros((2, 2*L**2), dtype=np.dtype('b'))
        for k in range(L):
            _flatXflips2Zerr[0, (2*k+1)%(2*L)*L+(0  )%L] = 1
            _flatXflips2Zerr[1, (2*0  )%(2*L)*L+(k  )%L] = 1
        return _flatXflips2Zerr

    @property
    def flatZflips2Xerr(self):
        L = self.L
        _flatZflips2Xerr = np.zeros((2, 2*L**2), dtype=np.dtype('b'))
        for k in range(L):
            _flatZflips2Xerr[0, (2*0+1)%(2*L)*L+(k  )%L] = 1
            _flatZflips2Xerr[1, (2*k  )%(2*L)*L+(0  )%L] = 1
        return _flatZflips2Xerr

    def H(self, Z=True, X=False):
        H = []
        if Z:
            H.append(self.flatXflips2Zstab)
        if X:
            H.append(self.flatZflips2Xstab)
        H = scipy.linalg.block_diag(*H)
        return H

    def E(self, Z=True, X=False):
        E = []
        if Z:
            E.append(self.flatXflips2Zerr)
        if X:
            E.append(self.flatZflips2Xerr)
        E = scipy.linalg.block_diag(*E)
        return E

##########################################################################################

class Honeycomb666Code:
    '''
    Honeycomb 6-6-6 code implementation on a triangle layout
    
    The honeycomb lattice is constructed as a hexagonal grid inside a triangle.
    - Physical qubits are placed on the vertices
    - Each hexagon (face) corresponds to a stabilizer generator (both for X and Z)
    - Logical operators are defined by an edge of the triangle

    Hexagonal lattice structure:

                   q0
                 /   \
                /     q1
               /     /  \
              q2 - q3    \
             /       \    \
            q4       q5 - q6
          /  \       /      \
         /    q7 - q8       q9
        /     /      \      /  \
       q10 - q11     q12 - q13  \
      /       \      /       \   \
    q14  - - q15 - q16 -  - q17 - q18

    The first level is qubit 0 (1 qubits)
    The second level is qubit 1 (1 qubits)
    The third level are qubits 2-3 (2 qubits)
    The fourth level are qubits 4-6 (3 qubits)
    The fifth level are qubits 7-9 (3 qubits)
    The sixth level are qubits 10-13 (4 qubits)
    The seventh level are qubit 14-18 (5 qubits)
    Where H is a hexagonal face (stabilizer) and q0-q18 are qubits on the edges.
    '''
    
    def __init__(self, L):
        '''
        Honeycomb 6-6-6 code with L qubits on one edge of the triangle.
        
        Args:
            L: Linear size parameter, number of qubits on one side of the triangle.
        '''
        assert(L % 2 == 1), "L must be odd"  # not sure how to handle even L
        self.L = L # number of qubits on one side of the triangle
        self.n_qubits = self._n_qubits_by_L(L) # total number of qubits
        self.n_level = self._n_levels_by_L(L) # number of levels in the triangle
        assert(self.n_qubits == self._n_qubits_until_level(self.n_level))

    @staticmethod
    def _n_qubits_by_L(L):
        return int(3/4 * (L**2) + 1/4)

    @staticmethod
    def _n_levels_by_L(L):
        return int(1.5 * L - 0.5)

    @lru_cache(maxsize=None)
    def _n_qubits_in_level(self, level):
        return (2 * level + 3) // 3

    @lru_cache(maxsize=None)
    def _n_qubits_until_level(self, level):
        return sum(self._n_qubits_in_level(l) for l in range(level))

    def _qubit_index(self, level, j):
        if j >= 0:
            return self._n_qubits_until_level(level) + j
        else:
            # j is negative, we are counting from the end of the level
            return self._n_qubits_until_level(level + 1) + j

    def _is_qubit_on_right_edge_in_level(self, level):
        return level % 3 != 1

    def _stabilizers(self):
        stabilizers = []
        # Red stabilizers
        for level in range(0, self.n_level - 2, 3):
            # stabilizer on the left edge
            stab = np.zeros((self.n_qubits,), dtype=np.dtype('b'))
            stab[self._qubit_index(level, 0)] = 1
            stab[self._qubit_index(level + 1, 0)] = 1
            stab[self._qubit_index(level + 2, 0)] = 1
            stab[self._qubit_index(level + 2, 1)] = 1
            stabilizers.append(stab)

            for i in range(1, self._n_qubits_in_level(level), 2):
                # stabilizer in the middle
                stab = np.zeros((self.n_qubits,), dtype=np.dtype('b'))
                stab[self._qubit_index(level, i)] = 1
                stab[self._qubit_index(level, i + 1)] = 1
                stab[self._qubit_index(level + 1, i)] = 1
                stab[self._qubit_index(level + 1, i + 1)] = 1
                stab[self._qubit_index(level + 2, i + 1)] = 1
                stab[self._qubit_index(level + 2, i + 2)] = 1
                stabilizers.append(stab)

        # Blue stabilizers
        for level in range(1, self.n_level - 2, 3):
            # stabilizer on the right edge
            stab = np.zeros((self.n_qubits,), dtype=np.dtype('b'))
            stab[self._qubit_index(level, -1)] = 1
            stab[self._qubit_index(level + 1, -1)] = 1
            stab[self._qubit_index(level + 2, -1)] = 1
            stab[self._qubit_index(level + 2, -2)] = 1
            stabilizers.append(stab)

            for i in range(0, self._n_qubits_in_level(level)-2, 2):
                # stabilizer in the middle
                stab = np.zeros((self.n_qubits,), dtype=np.dtype('b'))
                stab[self._qubit_index(level, i)] = 1
                stab[self._qubit_index(level, i + 1)] = 1
                stab[self._qubit_index(level + 1, i + 1)] = 1
                stab[self._qubit_index(level + 1, i + 2)] = 1
                stab[self._qubit_index(level + 2, i + 1)] = 1
                stab[self._qubit_index(level + 2, i + 2)] = 1
                stabilizers.append(stab)

        # Green stabilizers
        # bottom stabilizers
        for i in range(0, self._n_qubits_in_level(self.n_level - 2) - 1, 2):
            stab = np.zeros((self.n_qubits,), dtype=np.dtype('b'))
            stab[self._qubit_index(self.n_level - 1, i)] = 1
            stab[self._qubit_index(self.n_level - 1, i + 1)] = 1
            stab[self._qubit_index(self.n_level - 2, i)] = 1
            stab[self._qubit_index(self.n_level - 2, i + 1)] = 1
            stabilizers.append(stab)
        for level in range(2, self.n_level - 2, 3):
            for i in range(0, self._n_qubits_in_level(level), 2):
                # stabilizer in the middle
                stab = np.zeros((self.n_qubits,), dtype=np.dtype('b'))
                stab[self._qubit_index(level, i)] = 1
                stab[self._qubit_index(level, i + 1)] = 1
                stab[self._qubit_index(level + 1, i)] = 1
                stab[self._qubit_index(level + 1, i + 1)] = 1
                stab[self._qubit_index(level + 2, i)] = 1
                stab[self._qubit_index(level + 2, i + 1)] = 1
                stabilizers.append(stab)

        return np.array(stabilizers, dtype=np.dtype('b'))

    def _logical_operators(self):
        logical_operators = np.zeros((1, self.n_qubits), dtype=np.dtype('b'))
        for level in range(self.n_level):
            if self._is_qubit_on_right_edge_in_level(level):
                logical_operators[0, self._qubit_index(level, 0)] = 1
        return logical_operators

    def H(self, Z=True, X=False):
        stabilizers = self._stabilizers() # X and Z stabilizers are on the same qubits
        H = []
        if Z:
            H.append(stabilizers.copy())
        if X:
            H.append(stabilizers.copy())
        self.H = scipy.linalg.block_diag(*H)
        return self.H

    def E(self, Z=True, X=False):
        logical_operators = self._logical_operators() # X and Z logical operators are on the same qubits
        E = []
        if Z:
            E.append(logical_operators.copy())
        if X:
            E.append(logical_operators.copy())
        E = scipy.linalg.block_diag(*E)
        return E


##########################################################################################

def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

#############################################
def Get_toric_Code(L,full_H=False):
    toric = ToricCode(L)
    Hx = toric.H(Z=full_H,X=True)
    logX = toric.E(Z=full_H,X=True)    
    return Hx, logX

def Get_honeycomb_Code(L,full_H=False):
    honeycomb = Honeycomb666Code(L)
    Hx = honeycomb.H(Z=full_H,X=True)
    logX = honeycomb.E(Z=full_H,X=True)    
    return Hx, logX


#############################################
if __name__ == "__main__":
    Get_honeycomb_Code(5)
