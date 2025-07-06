import stim
import numpy as np


def get_error_array_format(circuit):
    """
    Get errors in format: [X0,X1,X2,X3,X4,X5,X6,Z0,Z1,Z2,Z3,Z4,Z5,Z6]
    Returns array of length 14 for each shot
    """

    # Sample with error data
    dem = circuit.detector_error_model()
    sampler = dem.compile_sampler()
    syndromes, obs_flips, err_data = sampler.sample(1, return_errors=True)
    explained = circuit.explain_detector_error_model_errors(
        dem_filter=dem,
        reduce_to_one_representative_error=False
    )

    error_array = np.zeros(14, dtype=int)
    for error_group in explained:
        for error in error_group:
            error_str = str(error)

            # Parse X errors (positions 0-6)
            if error_str.startswith('X'):
                qubit_idx = int(error_str[1:])
                if 0 <= qubit_idx <= 6:
                    error_array[qubit_idx] = 1

            # Parse Z errors (positions 7-13)
            elif error_str.startswith('Z'):
                qubit_idx = int(error_str[1:])
                if 0 <= qubit_idx <= 6:
                    error_array[7 + qubit_idx] = 1

            # Parse Y errors (both X and Z components)
            elif error_str.startswith('Y'):
                qubit_idx = int(error_str[1:])
                if 0 <= qubit_idx <= 6:
                    error_array[qubit_idx] = 1  # X component
                    error_array[7 + qubit_idx] = 1  # Z component

    return error_array


# Quick function to just get the error arrays
def get_error_qubits_array(circuit):
    """
    Simple function that returns just the error arrays
    Returns: numpy array of shape (num_shots, 14)
    """
    return get_error_array_format(circuit)