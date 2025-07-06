import stim
import torch
import random
import numpy as np
from torch.utils import data
from Codes import bin_to_sign, sign_to_bin
from torch.utils.data import DataLoader

from circuit_noise_errors import get_error_qubits_array


def setup_dataloader(code, noise_type, repetitions, seed, batch_size, workers, ps):
    return DataLoader(
        QECC_Dataset(
            code, ps, length=batch_size * 5000, noise_type=noise_type, repetitions=repetitions, seed=seed
            ),
            batch_size=int(batch_size), shuffle=True, num_workers=workers
    )

def setup_dataloader_list(code, noise_type, repetitions, seed, batch_size, workers, ps):
    return [
        DataLoader(
            QECC_Dataset(
                code, [p], length=int(batch_size), noise_type=noise_type, repetitions=repetitions, seed=seed
                ),
                batch_size=int(batch_size), shuffle=False, num_workers=workers)
        for p in ps
    ]


class QECC_Dataset(data.Dataset):
    """
    PyTorch Dataset for Quantum Error Correction Codes (QECC).
    Generates noisy codewords and corresponding syndromes for training/testing.
    """
    def __init__(self, code, ps, length, noise_type, repetitions, seed):
        """
        Args:
            code: Code object containing logic and parity-check matrices.
            ps: List or array of error probabilities.
            length: Number of samples in the dataset.
            noise_type: 'independent' or 'depolarization' or 'circuit'.
            repetitions: Number of faulty measurement rounds.
            seed: Random seed for reproducibility.
        """
        np.random.seed(seed)
        random.seed(seed)

        self.code = code
        self.ps = ps
        self.length = length
        self.logic_matrix = code.logic_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1).clone().cpu()
        self.zero_cw = torch.zeros((self.pc_matrix.shape[0])).long()
        self.noise_type = noise_type
        self.noise_method = {'independent': self.independent_noise, 'depolarization': self.depolarization_noise, 'circuit': self.circuit_noise}[noise_type]
        self.repetitions = repetitions
        self.circuit = None
        if noise_type == 'circuit':
            assert code.code_type == 'honeycomb', "Circuit noise is only implemented for honeycomb code."
            from color_code_circuit import TriangleColorCode
            self.circuit = TriangleColorCode(distance=code.distance)

    def independent_noise(self, pp=None):
        """Generate independent bit-flip noise."""
        pp = random.choice(self.ps) if pp is None else pp
        return np.random.binomial(1, pp, self.pc_matrix.shape[0])

    def depolarization_noise(self, pp=None):
        """Generate depolarizing noise (see neural-decoder reference)."""
        pp = random.choice(self.ps) if pp is None else pp
        out_dimZ = out_dimX = self.pc_matrix.shape[0] // 2

        def makeflips(q):
            q = q / 3.
            flips = np.zeros((out_dimZ + out_dimX,), dtype=np.dtype('b'))
            rand = np.random.rand(out_dimZ or out_dimX)
            both_flips = (2 * q <= rand) & (rand < 3 * q)
            x_flips = rand < q
            flips[:out_dimZ] ^= x_flips
            flips[:out_dimZ] ^= both_flips
            z_flips = (q <= rand) & (rand < 2 * q)
            flips[out_dimZ:out_dimZ + out_dimX] ^= z_flips
            flips[out_dimZ:out_dimZ + out_dimX] ^= both_flips
            return flips

        flips = makeflips(pp)
        while not np.any(flips):
            flips = makeflips(pp)
        return flips * 1.
    
    def circuit_noise(self, pp=None):
        """Generate noise based on a circuit model."""
        if self.circuit is None:
            raise ValueError("Circuit noise requires a circuit model to be defined.")
        pp = random.choice(self.ps) if pp is None else pp
        stim_circuit = self.circuit.get_circuit(num_cycle=self.repetitions-1, physical_error_rate=pp)
        err_data = get_error_qubits_array(stim_circuit)
        #syndromes, obs_flips, err_data = sampler.sample(1, return_errors=True)
        return err_data

    def _circuit_noise_get_item(self):
        """Handle the case for circuit noise."""
        pp = random.choice(self.ps)
        err_data = self.circuit_noise(pp=pp)
        z = torch.from_numpy(err_data)
        x = self.zero_cw
        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        syndrome = bin_to_sign(torch.matmul(z.long(), self.pc_matrix) % 2)
        return (
            x.float(),
            z.float(),
            y.float(),
            (magnitude * 0 + 1).float(),
            syndrome.float()
        )

    def __getitem__(self, index):
        if self.noise_type == 'circuit':
            # Circuit noise requires a different handling
            if self.circuit is None:
                raise ValueError("Circuit noise requires a circuit model to be defined.")
            return self._circuit_noise_get_item()
        x = self.zero_cw
        pp = random.choice(self.ps)
        if self.repetitions <= 1:
            z = torch.from_numpy(self.noise_method(pp))
            y = bin_to_sign(x) + z
            magnitude = torch.abs(y)
            syndrome = torch.matmul(z.long(), self.pc_matrix) % 2
            syndrome = bin_to_sign(syndrome)
            return (
                x.float(),
                z.float(),
                y.float(),
                (magnitude * 0 + 1).float(),
                syndrome.float()
            )
        # Multiple repetitions (faulty measurements)
        qq = pp
        noise_new = np.stack([self.noise_method(pp) for _ in range(self.repetitions)], 1)
        noise_cumulative = (np.cumsum(noise_new, 1) % 2).astype(np.uint8)
        noise_total = noise_cumulative[:, -1]
        syndrome = (
            torch.matmul(torch.from_numpy(noise_cumulative).long().transpose(0, 1), self.pc_matrix) % 2
        ).transpose(0, 1).numpy()
        syndrome_error = (np.random.rand(self.pc_matrix.shape[1], self.repetitions) < qq).astype(np.uint8)
        syndrome_error[:, -1] = 0  # Perfect measurements in last round
        noisy_syndrome = (syndrome + syndrome_error) % 2
        # Convert to difference syndrome
        noisy_syndrome[:, 1:] = (noisy_syndrome[:, 1:] - noisy_syndrome[:, 0:-1]) % 2

        z = torch.from_numpy(noise_total)
        syndrome = bin_to_sign(torch.from_numpy(noisy_syndrome))  # TODO: check if bin2sign is needed
        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        return (
            x.float(),
            z.float(),
            (y * 0 + 1).float(),
            (magnitude * 0 + 1).float(),
            syndrome.float().transpose(0, 1)
        )

    def __len__(self):
        return self.length