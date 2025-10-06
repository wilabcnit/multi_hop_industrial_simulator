"""
    Utility scripts tha computes the antenna gain as proposed in
    Vijay Kumar Salvia, Antenna and wave propagation (Ed. Laxmi, 2007)
"""

import numpy as np
from math import log10
from scipy import constants as ct

# Function to compute the antenna gain in dB given the number of antenna elements
def get_antenna_gain_db(number_of_antennas: int):
    """

    Args:
      number_of_antennas: int: 

    Returns:

    """

    gain = 0
    if number_of_antennas == 1:
        gain = 1
    elif number_of_antennas < 1:
        exit("The number of antenna elements cannot be lower than 1")
    else:
        theta = 2 * np.arcsin(2 / number_of_antennas)
        gain = 41000 / pow((theta * 360 / (2 * ct.pi)), 2)

    gain_db = 10 * log10(gain)

    return gain_db
