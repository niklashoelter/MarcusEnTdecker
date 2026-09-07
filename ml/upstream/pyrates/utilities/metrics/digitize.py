import numpy as np

def digitize_spin_populations(spin_populations: np.ndarray, bins: list = [-1, 0.07, 0.2, 0.4, 0.8, 2]):
    '''
    Function to digitize spin populations based on specified bins.
    :param spin_populations: array of spin population values to be digitized.
    :param bins: list of bin edges for digitization. Default is [-1, 0.07, 0.2, 0.4, 0.8, 2].
    :return: array of digitized values corresponding to the input spin populations.
    '''
    bins = np.array(bins)
    digitized = np.digitize(spin_populations, bins)
    return digitized