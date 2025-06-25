"""
Implementation of "Deep Quantum Error Correction" (DQEC), AAAI24
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
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
    Honeycomb 6-6-6 code implementation on a closed (torus) layout
    
    The honeycomb lattice is constructed as a hexagonal grid where:
    - Physical qubits are placed on the vertices
    - Each hexagon (face) corresponds to a stabilizer generator
    
    Hexagonal lattice structure:
           q0 - q1
          /       \
        q5    H    q2
          \       /
           q4 - q3
    
    Where H is a hexagonal face (stabilizer) and q0-q5 are qubits on the edges.

    Each hexagon is a stabilizer for both X and Z errors.
    '''
    
    def __init__(self, L):
        '''
        Honeycomb 6-6-6 code with L hexagons on one side of the lattice
        
        Args:
            L: Linear size parameter (actual number of qubits will be larger)
        '''
        self.L = L # number of hexagons on one side of the lattice
        self.W = (L + 1) // 2 * 2 # width of the lattice (even number)
        self.n_qubits = 2 * self.L * self.W # total number of qubits
        self.n_hexagons = self.L * self.W # total number of hexagons

    @property
    def flatXflips2Zstab(self):
        stab = np.zeros((self.n_hexagons, self.n_qubits), dtype=np.dtype('b'))
        for i, j in itertools.product(range(self.L),range(self.W)):
            # i is the row, j is the column
            # on an even column (starting from 0), the first edge of the hexagon is (2*i, j)
            # and on an odd column, the first edge is (2*i+1,j)
            start_row = 2 * i if j%2 == 0 else 2*i+1
            middle_row = (start_row + 1) % (2 * self.L)
            end_row = (start_row + 2) % (2 * self.L)
            start_column = j
            end_column = (j + 1) % self.W
            stab[i*self.W+j, start_row * self.W + start_column] = 1
            stab[i*self.W+j, middle_row * self.W + start_column] = 1
            stab[i*self.W+j, end_row * self.W + start_column] = 1
            stab[i*self.W+j, start_row * self.W + end_column] = 1
            stab[i*self.W+j, middle_row * self.W + end_column] = 1
            stab[i*self.W+j, end_row * self.W + end_column] = 1
        return stab

    @property
    def flatZflips2Xstab(self):
        stab = np.zeros((self.n_hexagons, self.n_qubits), dtype=np.dtype('b'))
        for i, j in itertools.product(range(self.L),range(self.W)):
            # i is the row, j is the column
            # on an even column (sarting from 0), the first edge of the hexagon is (2*i, j)
            # and on an odd column, the first edge is (2*i+1,j)
            start_row = 2 * i if j%2 == 0 else 2*i+1
            middle_row = (start_row + 1) % (2 * self.L)
            end_row = (start_row + 2) % (2 * self.L)
            start_column = j
            end_column = (j + 1) % self.W
            stab[i*self.W+j, start_row * self.W + start_column] = 1
            stab[i*self.W+j, middle_row * self.W + start_column] = 1
            stab[i*self.W+j, end_row * self.W + start_column] = 1
            stab[i*self.W+j, start_row * self.W + end_column] = 1
            stab[i*self.W+j, middle_row * self.W + end_column] = 1
            stab[i*self.W+j, end_row * self.W + end_column] = 1
        return stab

    @property
    def flatXflips2Zerr(self):
        _flatXflips2Zerr = np.zeros((2, self.n_qubits), dtype=np.dtype('b'))
        for k in range(2 * self.L):  # vertical circle
            _flatXflips2Zerr[0, k * self.W] = 1
        for k in range(self.W): # horizontal circle
            _flatXflips2Zerr[1, k] = 1
            _flatXflips2Zerr[1, self.W + k] = 1
        return _flatXflips2Zerr

    @property
    def flatZflips2Xerr(self):
        L = self.L
        _flatZflips2Xerr = np.zeros((2, self.n_qubits), dtype=np.dtype('b'))
        for k in range(self.W): # horizontal circle
            _flatZflips2Xerr[0, k] = 1
            _flatZflips2Xerr[1, self.W + k] = 1
        for k in range(2 * self.L): # vertical circle
            _flatZflips2Xerr[1, k * self.W] = 1
        return _flatZflips2Xerr

    def H(self, Z=True, X=False):
        H = []
        if Z:
            H.append(self.flatXflips2Zstab)
        if X:
            H.append(self.flatZflips2Xstab)
        self.H = scipy.linalg.block_diag(*H)
        return self.H

    def E(self, Z=True, X=False):
        E = []
        if Z:
            E.append(self.flatXflips2Zerr)
        if X:
            E.append(self.flatZflips2Xerr)
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
    Get_honeycomb_Code(4)
    class Code:
        pass
